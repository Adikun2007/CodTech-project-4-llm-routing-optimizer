# After optimizer runs, this file:

# Calculates the naive baseline cost (everything routed to GPT-5.5)
# Compares naive vs optimized
# Prints the savings amount and percentage
# Generates 4 visualizations using matplotlib

# analysis.py — Business Impact Analysis + Visualizations
# WHAT THIS FILE DOES:
#   optimizer.py solved the math and saved assignments.json
#   This file loads that result and answers the business question:
#   "How much did we actually save, and what does the routing look like?"
#
# OUTPUT:
#   - Printed cost breakdown in terminal
#   - routing_analysis.png with 4 charts saved to data/

import pandas as pd
import json
import matplotlib.pyplot as plt

# STEP 1 — LOAD RESULTS FROM optimizer.py
# optimizer.py saved everything into assignments.json
# We load it here so analysis.py is independent — you can run it anytime
# without re-solving the LP

with open('data/assignments.json', 'r') as f:
    results = json.load(f)

# pulling out the key pieces of info we need for analysis:
assignments = results['assignments']  # which query type assigned to which model
optimized_cost = results['total_optimized_cost']  # total cost after optimization
baseline_cost = results['baseline_cost']  # total cost if everything went to GPT-5.5
savings = results['savings']  # how much we saved in dollars
savings_pct = results['savings_pct']  # how much we saved in percentage

# Also load CSVs for extra info (model colors, query volumes etc.)
models_df  = pd.read_csv("data/model_data.csv")
queries_df = pd.read_csv("data/query_data.csv")


# STEP 2 — ASSIGN A COLOR TO EACH MODEL

model_colors = {
    "GPT-5.5":               "#10a37f",   # OpenAI green
    "Claude Sonnet 4.5":     "#d97706",   # Anthropic amber
    "Gemini 3.1 Flash-Lite": "#4285F4",   # Google blue
    "Llama 3 70B":           "#7c3aed",   # Meta purple
    "Mistral 7B":            "#e11d48",   # Mistral red
}


# STEP 3 — BUILD A CLEAN DATAFRAME FROM ASSIGNMENTS
# Each row = one query type with all its details
rows = []
for query, info in assignments.items():
    rows.append({
        "query_type":    query,
        "model":         info["model"],
        "monthly_cost":  info["monthly_cost"],
        "volume":        info["volume"],
        "avg_tokens":    info["avg_tokens"],
        "quality_score": info["quality_score"],
        "min_quality":   info["min_quality"],
        # naive cost = what this query would cost if routed to GPT-5.5
        "naive_cost":    round(info["volume"] * info["avg_tokens"] / 1000 * 0.005, 4)
    })

df = pd.DataFrame(rows)

# Sort by optimized monthly cost descending — most expensive queries on top
df = df.sort_values("monthly_cost", ascending=False).reset_index(drop=True)


# STEP 4 — PRINT BUSINESS IMPACT SUMMARY TO TERMINAL

print("BUSINESS IMPACT SUMMARY")
print("=" * 65)
print(f"  Naive monthly cost (all GPT-5.5) : ${baseline_cost:.2f}")
print(f"  Optimized monthly cost           : ${optimized_cost:.2f}")
print(f"  Monthly savings                  : ${savings:.2f}")
print(f"  Savings %                        : {savings_pct}%")
print(f"  Annual projected savings         : ${savings * 12:.2f}")
print("=" * 65)

print("\nPER QUERY TYPE BREAKDOWN:")
print(f"{'Query Type':<30} {'Model':<25} {'Naive $':<10} {'Optimized $':<12} {'Saved $'}")
print("-" * 90)
for _, row in df.iterrows():
    saved = row["naive_cost"] - row["monthly_cost"]
    print(f"{row['query_type']:<30} {row['model']:<25} ${row['naive_cost']:<9.4f} ${row['monthly_cost']:<11.4f} ${saved:.4f}")

# STEP 5 — CREATE THE FIGURE (4 CHARTS IN ONE IMAGE)
