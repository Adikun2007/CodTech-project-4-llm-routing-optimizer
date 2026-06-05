# Step 2 — optimizer.py (the core file)
# This is your main file. It does 4 things in order:

# Load both CSVs using pandas
# Build the LP problem in PuLP — define decision variables, objective function, constraints
# Solve it — PuLP calls the solver, gets optimal assignments
# Print results — which query type goes to which model, total optimized cost

# The Math First
# What you're solving:
# You have 10 query types and 5 models. For every query type, you need to pick exactly 1 model. That's your decision.
# Decision Variable:
# x[i][j] = 1  if query type i is assigned to model j
# x[i][j] = 0  if not
# So you have 10 × 5 = 50 binary variables (0 or 1 only). This makes it a Binary Integer Linear Program (BILP).
# Objective Function — what you're minimizing:
# Cost = volume[i] × avg_tokens[i] × cost_per_1k[j] × x[i][j]  ÷ 1000
# Summed across all i (query types) and j (models).
# In plain english: for each assignment, monthly cost = how many queries × how long each query × price per token of that model.
# Two Constraints:
# Constraint 1 — Each query type must go to exactly one model:
# Σ x[i][j] = 1   for every query type i
# Constraint 2 — Assigned model must meet quality requirement:
# x[i][j] = 0   if quality_score[j] < min_quality[i]
# This is enforced before solving by simply not creating those variables.


# optimizer.py — LLM Query Routing Optimizer
# =============================================================================
# BUSINESS PROBLEM:
#   A company runs an AI chatbot that handles multiple query types.
#   They currently send everything to GPT-5.5 (most expensive model).
#   Goal: Route each query type to the CHEAPEST model that still meets
#         the minimum quality requirement for that query.
#
# MATH PROBLEM TYPE: Binary Integer Linear Program (BILP)
#   - Binary because x[i][j] is either 0 or 1 (assigned or not)
#   - Integer because no fractional assignments (can't assign 0.5 of a query)
#   - Linear because objective function and constraints are all linear

import pandas as pd
from pulp import (
    LpProblem,       # creates the LP problem object
    LpMinimize,      # tells PuLP we want to minimize (not maximize)
    LpVariable,      # creates decision variables
    lpSum,           # sums a list of LP expressions (like sum() but for PuLP)
    LpBinary,        # makes variables binary (0 or 1 only)
    value,           # extracts the solved numerical value from a variable
    LpStatus,        # converts status code to readable string
    PULP_CBC_CMD     # the solver engine (CBC = open source, free, built-in)
)

# Step 1: Load Data
# Load query data and model data from CSV files into pandas DataFrames
model_df = pd.read_csv('data/model_data.csv')
query_df = pd.read_csv('data/query_data.csv')

# converting data frames to lists for easier access in PuLP
model_names = model_df['model_name'].tolist()
query_types   = query_df["query_type"].tolist()

# convert model attributes to dictionaries for easy lookup
# Example: cost["GPT-5.5"] = 0.005
cost = dict(zip(model_df['model_name'], model_df['cost_per_1k_tokens']))
quality = dict(zip(model_df['model_name'], model_df['quality_score']))
max_query = dict(zip(model_df['model_name'], model_df['max_monthly_queries']))

volume        = dict(zip(query_df["query_type"], query_df["monthly_volume"]))
avg_tokens    = dict(zip(query_df["query_type"], query_df["avg_tokens"]))
min_quality   = dict(zip(query_df["query_type"], query_df["min_quality_required"]))

print("=" * 100)
print("DATA LOADED SUCCESSFULLY")
print(f"  Models loaded  : {len(model_names)}")
print(f"  Query types    : {len(query_types)}")
print(f"  Decision vars  : {len(model_names) * len(query_types)} (binary)")
print("=" * 100)


# STEP 2 — CREATE THE LP PROBLEM
# LpProblem() creates an empty container for your optimization problem.
# You give it:
#   - a name (just a label, doesn't affect solving)
#   - a sense: LpMinimize (we want the lowest cost)
#
# Think of it like an empty equation sheet — you'll fill it next.

# Create the LP problem object
problem = LpProblem("LLM_Query_ROuting_Optimization", LpMinimize)


# STEP 3 — DECISION VARIABLES
# x[i][j] = 1 if query type i is assigned to model j, else 0
#
# We use a nested dict: x["Code Generation"]["GPT-5.5"] = LpVariable(...)
#
# LpVariable() creates one binary variable. Parameters:
#   - name: must be unique string (we use f"x_{i}_{j}" with spaces removed)
#   - cat=LpBinary: forces variable to be 0 or 1 (not 0.7, not 2)
#
# IMPORTANT OPTIMIZATION HERE:
#   We only create x[i][j] if model j CAN serve query i (quality check).
#   If quality[j] < min_quality[i], that assignment is impossible anyway,
#   so we don't even create that variable — saves computation.
#   We force it to 0 using a regular Python float instead of LpVariable.

# Create decision variables
x = {} # outer dict: keys = query types
for i in query_types:
    x[i] = {} # inner dict: keys = model names
    for j in model_names:
        if quality[j] >= min_quality[i]:
            # This model CAN handle this query — create a real binary variable
            var_name = f"x_{i}_{j}".replace(" ", "_")  # unique name for variable
            x[i][j] = LpVariable(var_name, cat=LpBinary)  # binary variable
        else:
            x[i][j] = 0  # not a variable, just a fixed value (can't assign)

real_vars_count = sum(
    1 for i in query_types for j in model_names 
    if isinstance(x[i][j], type(LpVariable("test")))
)
print(f"\nDecision variables after quality filtering: {real_vars_count}")
print(f"Variables eliminated by quality constraint: {len(query_types)*len(model_names) - real_vars_count}")


# STEP 4 — DEFINE OBJECTIVE FUNCTION
# This is what PuLP will MINIMIZE.
#
# COST FORMULA for one assignment (query i → model j):
#   monthly_cost = volume[i] × avg_tokens[i] × cost_per_1k[j] / 1000 × x[i][j]
#
# Breaking it down:
#   volume[i]          = how many queries per month (e.g. 5000)
#   avg_tokens[i]      = average tokens per query (e.g. 200)
#   volume × tokens    = total tokens used per month (e.g. 1,000,000)
#   ÷ 1000             = convert to "per 1K tokens" units
#   × cost_per_1k[j]   = multiply by model's price (e.g. $0.005)
#   × x[i][j]          = if x=0, this term vanishes. if x=1, it's included.
#
# lpSum([...]) adds all these terms together into one expression.
# We pass this expression to prob += which sets it as the objective.

problem += lpSum(
    volume[i] * avg_tokens[i] * cost[j] / 1000 * x[i][j]
    for i in query_types for j in model_names
), "Total_Monthly_Cost"


# STEP 5 — ADD CONSTRAINTS

# --- CONSTRAINT 1: Each query type must be assigned to EXACTLY 1 model ---
#
# For every query type i:
#   x[i]["GPT-5.5"] + x[i]["Claude"] + x[i]["Gemini"] + ... = 1
#
# This ensures:
#   - No query type is left unassigned (sum can't be 0)
#   - No query type is split across models (sum can't be 2)
#
# The loop adds one constraint per query type (10 constraints total).
for i in query_types:
    problem += lpSum(x[i][j]for j in model_names) == 1, f"OneModelPerQuery_{i.replace(' ', '_')}"

# --- CONSTRAINT 2: No model exceeds its monthly query capacity ---
#
# For every model j:
#   sum of (volume[i] × x[i][j]) across all query types ≤ max_queries[j]
#
# This means: total queries routed to model j can't exceed its limit.
# (We set all limits to 50000 so this won't bind, but it's good practice
#  to include capacity constraints — shows real-world thinking.)
for j in model_names:
    problem += lpSum(volume[i] * x[i][j] for i in query_types) <= max_query[j], f"Capacity_{j.replace(' ', '_')}"

print(f"\nConstraints added:")
print(f"  Assignment constraints : {len(query_types)} (one per query type)")
print(f"  Capacity constraints   : {len(model_names)} (one per model)")


# STEP 6 — SOLVE
# =============================================================================
# PULP_CBC_CMD() is the solver engine.
#   msg=0 means: don't print solver logs (keeps output clean)
#
# problem.solve() triggers the Branch-and-Bound algorithm internally:
#   1. Relaxes binary constraint (allows 0 to 1 continuous values)
#   2. Solves the relaxed LP (fast)
#   3. Branches on fractional variables, forcing them to 0 or 1
#   4. Repeats until all variables are integer
#
# This finds the GLOBALLY OPTIMAL solution (not just a good one).

print("\nSolving the optimization problem...")
problem.solve(PULP_CBC_CMD(msg=0))  # silent solver

# Check if a valid solution was found
print(f"Status: {LpStatus[problem.status]}")

if problem.status != 1:
    print("No optimal solution found. Check your constraints.")
    exit()


# STEP 7 — EXTRACT AND DISPLAY RESULTS
# value(x[i][j]) extracts the solved value (0.0 or 1.0) from each variable.
# We look for where value == 1.0 — that's the assigned model.

print("OPTIMAL ROUTING ASSIGNMENTS")
print(f"{'Query Type':<30} {'Assigned Model':<25} {'Monthly Cost ($)'}")
print("-" * 70)

total_optimized_cost = 0
assignments = {}  # store for use in analysis.py later

for i in query_types:
    for j in model_names:
        if value(x[i][j]) == 1:
            # Calculate the actual monthly cost for this assignment
            monthly_cost = volume[i] * avg_tokens[i] / 1000 * cost[j]
            total_optimized_cost += monthly_cost
            assignments[i] = {
                "model": j,
                "monthly_cost": round(monthly_cost, 4),
                "volume": volume[i],
                "avg_tokens": avg_tokens[i],
                "quality_score": quality[j],
                "min_quality": min_quality[i]
            }
            print(f"{i:<30} {j:<25} ${monthly_cost:.4f}")


# STEP 8 — BASELINE COMPARISON
# Naive approach: route EVERYTHING to GPT-5.5 (most expensive model).
# This is what a company does without optimization.
# We compare this to our optimized cost to show business impact.

most_expensive_model = max(cost, key=cost.get)  # finds model with highest cost
baseline_cost = sum(
    volume[i] * avg_tokens[i] / 1000 * cost[most_expensive_model]
    for i in query_types
)

savings = baseline_cost - total_optimized_cost
savings_pct = (savings / baseline_cost) * 100

print("\n" + "=" * 100)
print("COST SUMMARY")
print("=" * 100)
print(f"  Naive cost (all {most_expensive_model})         : ${baseline_cost:.2f}/month")
print(f"  Optimized cost                   : ${total_optimized_cost:.2f}/month")
print(f"  Monthly savings                  : ${savings:.2f}")
print(f"  Savings percentage               : {savings_pct:.1f}%")
print(f"  Annual savings                   : ${savings * 12:.2f}")
print("=" * 100)


# Save assignments dict so analysis.py can import it
import json
with open("data/assignments.json", "w") as f:
    json.dump({
        "assignments": assignments,
        "total_optimized_cost": round(total_optimized_cost, 4),
        "baseline_cost": round(baseline_cost, 4),
        "savings": round(savings, 4),
        "savings_pct": round(savings_pct, 2)
    }, f, indent=2)

print("\nassignments.json saved ready for analysis.py")