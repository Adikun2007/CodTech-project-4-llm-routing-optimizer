# =============================================================================
# analysis.py — Charts and Business Impact
# =============================================================================
# This file loads the results from optimizer.py and makes 4 simple charts.
# Each chart is written separately so you can clearly see what each one does.
# =============================================================================

import json
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1 — LOAD THE RESULTS
# =============================================================================
# optimizer.py already solved everything and saved assignments.json
# We just load that file here — no math, just reading data

with open("data/assignments.json", "r") as f:
    results = json.load(f)

assignments    = results["assignments"]
optimized_cost = results["total_optimized_cost"]
baseline_cost  = results["baseline_cost"]
savings        = results["savings"]
savings_pct    = results["savings_pct"]

# =============================================================================
# STEP 2 — CONVERT TO A SIMPLE TABLE (DataFrame)
# =============================================================================
# We turn the assignments dict into a simple table so it's easy to work with
# Each row = one query type

rows = []
for query, info in assignments.items():
    rows.append({
        "query_type":   query,
        "model":        info["model"],
        "opt_cost":     info["monthly_cost"],
        "naive_cost":   round(info["volume"] * info["avg_tokens"] / 1000 * 0.005, 4),
        "volume":       info["volume"],
        "quality":      info["quality_score"],
        "min_quality":  info["min_quality"]
    })

df = pd.DataFrame(rows)

# Calculate how much each query type saved
df["saved"] = df["naive_cost"] - df["opt_cost"]

# =============================================================================
# STEP 3 — PRINT SUMMARY IN TERMINAL
# =============================================================================

print("=" * 55)
print("COST SUMMARY")
print("=" * 55)
print(f"  Naive cost (all GPT-5.5) : ${baseline_cost:.2f}/month")
print(f"  Optimized cost           : ${optimized_cost:.2f}/month")
print(f"  Monthly savings          : ${savings:.2f}")
print(f"  Savings %                : {savings_pct}%")
print(f"  Annual savings           : ${savings * 12:.2f}")
print("=" * 55)

# =============================================================================
# STEP 4 — CHART COLORS
# =============================================================================
# Each model gets one fixed color — used consistently across all charts

model_colors = {
    "GPT-5.5":               "#10a37f",
    "Claude Sonnet 4.5":     "#f59e0b",
    "Gemini 3.1 Flash-Lite": "#3b82f6",
    "Llama 3 70B":           "#8b5cf6",
    "Mistral 7B":            "#ef4444",
}

# =============================================================================
# STEP 5 — CREATE THE FIGURE WITH 4 CHARTS
# =============================================================================
# fig is the whole image
# ax1, ax2, ax3, ax4 are the 4 individual chart boxes inside it

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("LLM Routing Optimizer — Cost Analysis", fontsize=15,
             fontweight="bold", color="white", y=0.99)

# -------------------------------------------------------
# CHART 1 — How much does each query type cost?
#           Two bars side by side: naive vs optimized
# -------------------------------------------------------
# Left bar  = what it would cost routing to GPT-5.5 (naive)
# Right bar = what it actually costs after optimization
# The gap between the two bars = money saved

queries    = df["query_type"].tolist()        # list of query names for x-axis
n          = len(queries)                     # how many queries (10)
x          = list(range(n))                  # [0,1,2,...,9] — positions on x-axis
bar_width  = 0.35                             # how wide each bar is

# Draw naive bars (left side of each group)
ax1.bar([i - bar_width/2 for i in x], df["naive_cost"],
        width=bar_width, color="#10a37f", alpha=0.5, label="Naive (GPT-5.5)")

# Draw optimized bars (right side of each group) — color by model
opt_colors = [model_colors[m] for m in df["model"]]
ax1.bar([i + bar_width/2 for i in x], df["opt_cost"],
        width=bar_width, color=opt_colors, alpha=0.9, label="Optimized")

ax1.set_xticks(x)
ax1.set_xticklabels(queries, rotation=35, ha="right", fontsize=7.5, color="white")
ax1.set_ylabel("Monthly Cost ($)", color="white")
ax1.set_title("Naive vs Optimized Cost per Query Type", color="white", fontweight="bold")
ax1.set_facecolor("#1a1a2e")
ax1.tick_params(colors="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# -------------------------------------------------------
# CHART 2 — Which model handles the most queries? (Pie)
# -------------------------------------------------------
# Group all queries by their assigned model
# Sum up the monthly volumes per model
# Each slice = one model, sized by how many queries it handles

volume_by_model = df.groupby("model")["volume"].sum()
pie_colors = [model_colors[m] for m in volume_by_model.index]

ax2.pie(
    volume_by_model.values,
    labels=volume_by_model.index,
    colors=pie_colors,
    autopct="%1.1f%%",        # show % on each slice
    startangle=140,
    wedgeprops={"edgecolor": "#0f0f0f", "linewidth": 2}
)

ax2.set_title("Query Volume Distribution by Model", color="white", fontweight="bold")
ax2.set_facecolor("#1a1a2e")
# Style the labels
for text in ax2.texts:
    text.set_color("white")
    text.set_fontsize(8)

# -------------------------------------------------------
# CHART 3 — How much does each model cost per month?
# -------------------------------------------------------
# Group by model, sum all costs for queries assigned to that model
# Horizontal bars so model names are easy to read

cost_by_model  = df.groupby("model")["opt_cost"].sum().sort_values()
bar_clrs       = [model_colors[m] for m in cost_by_model.index]

bars = ax3.barh(cost_by_model.index, cost_by_model.values,
                color=bar_clrs, height=0.5)

# Add $ label at end of each bar
for bar, val in zip(bars, cost_by_model.values):
    ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"${val:.2f}", va="center", color="white", fontsize=9)

ax3.set_xlabel("Monthly Cost ($)", color="white")
ax3.set_title("Monthly Cost Contribution per Model", color="white", fontweight="bold")
ax3.set_facecolor("#1a1a2e")
ax3.tick_params(colors="white")

# -------------------------------------------------------
# CHART 4 — How much did each query type save?
# -------------------------------------------------------
# Simple bar chart — one bar per query type
# Bar height = how many dollars were saved vs naive GPT-5.5 routing

df_sorted  = df.sort_values("saved", ascending=False)
bar_clrs4  = [model_colors[m] for m in df_sorted["model"]]

ax4.bar(df_sorted["query_type"], df_sorted["saved"],
        color=bar_clrs4, edgecolor="white", linewidth=0.5)

# Add $ label on top of each bar
for i, val in enumerate(df_sorted["saved"]):
    ax4.text(i, val + 0.05, f"${val:.2f}", ha="center",
             color="white", fontsize=7.5)

ax4.set_xticklabels(df_sorted["query_type"], rotation=35, ha="right",
                    fontsize=7.5, color="white")
ax4.set_ylabel("Savings ($)", color="white")
ax4.set_title("Savings per Query Type vs Naive Routing", color="white", fontweight="bold")
ax4.set_facecolor("#1a1a2e")
ax4.tick_params(colors="white")
ax4.set_xticks(range(len(df_sorted)))

# =============================================================================
# STEP 6 — ADD SAVINGS BANNER AT BOTTOM OF IMAGE
# =============================================================================

fig.text(0.5, 0.01,
         f"Monthly Savings: ${savings:.2f}  |  Annual Savings: ${savings*12:.2f}  |  Cost Reduced by {savings_pct}%",
         ha="center", fontsize=11, color="#00ff88", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                   edgecolor="#00ff88", linewidth=1.5))

# =============================================================================
# STEP 7 — SAVE THE IMAGE
# =============================================================================

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("data/routing_analysis.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f0f")
plt.close()

print("\nrouting_analysis.png saved to data/")
print("Done.")
