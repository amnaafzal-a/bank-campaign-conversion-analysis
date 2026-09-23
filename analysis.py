"""
Bank Campaign Conversion Analysis
Dataset: UCI Bank Marketing (Portuguese bank, 4,521 customers)
Tools: pandas, matplotlib
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("bank_raw.csv", sep=";", quotechar='"')
df["Converted"] = (df["y"] == "yes").astype(int)

overall_rate = df["Converted"].mean() * 100
print(f"Overall conversion rate: {overall_rate:.1f}% (n={len(df)})")

# ---------- Q1: Conversion rate by job/segment ----------
q1 = df.groupby("job")["Converted"].agg(["mean", "count"]).reset_index()
q1["conv_rate_pct"] = (q1["mean"] * 100).round(1)
q1 = q1.sort_values("conv_rate_pct", ascending=False)
print("\n--- Q1: Conversion rate by occupation ---")
print(q1[["job", "conv_rate_pct", "count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2E7D32" if v > overall_rate else "#90A4AE" for v in q1["conv_rate_pct"]]
ax.barh(q1["job"], q1["conv_rate_pct"], color=colors)
ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1, label=f"Overall avg ({overall_rate:.1f}%)")
ax.set_title("Students & Retirees Convert Nearly 3x the Average Rate", fontsize=12, fontweight="bold")
ax.set_xlabel("Conversion Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig("chart1_conversion_by_job.png", dpi=150)
plt.close()

# ---------- Q2: Contact frequency vs conversion (diminishing returns) ----------
def campaign_band(c):
    if c == 1: return "1 contact"
    if c <= 3: return "2-3 contacts"
    if c <= 6: return "4-6 contacts"
    return "7+ contacts"

df["ContactBand"] = df["campaign"].apply(campaign_band)
order = ["1 contact", "2-3 contacts", "4-6 contacts", "7+ contacts"]
q2 = df.groupby("ContactBand")["Converted"].agg(["mean", "count"]).reindex(order).reset_index()
q2["conv_rate_pct"] = (q2["mean"] * 100).round(1)
print("\n--- Q2: Conversion rate by number of contacts ---")
print(q2[["ContactBand", "conv_rate_pct", "count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(q2["ContactBand"], q2["conv_rate_pct"], marker="o", linewidth=2.5, color="#1565C0")
for i, v in enumerate(q2["conv_rate_pct"]):
    ax.annotate(f"{v}%", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
ax.set_title("More Contact Attempts = Sharply Diminishing Returns", fontsize=12, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.set_xlabel("Number of Contact Attempts")
plt.tight_layout()
plt.savefig("chart2_conversion_by_contacts.png", dpi=150)
plt.close()

# ---------- Q3: Does prior campaign outcome predict future success? ----------
q3 = df.groupby("poutcome")["Converted"].agg(["mean", "count"]).reset_index()
q3["conv_rate_pct"] = (q3["mean"] * 100).round(1)
q3 = q3.sort_values("conv_rate_pct", ascending=False)
print("\n--- Q3: Conversion rate by previous campaign outcome ---")
print(q3[["poutcome", "conv_rate_pct", "count"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#2E7D32", "#90A4AE", "#C62828", "#90A4AE"]
ax.bar(q3["poutcome"], q3["conv_rate_pct"], color=colors[:len(q3)])
for i, v in enumerate(q3["conv_rate_pct"]):
    ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
ax.set_title("Past Success Is the Strongest Predictor of Future Conversion", fontsize=12, fontweight="bold")
ax.set_ylabel("Conversion Rate (%)")
ax.set_xlabel("Outcome of Previous Campaign")
plt.tight_layout()
plt.savefig("chart3_conversion_by_history.png", dpi=150)
plt.close()

# ---------- Q4: Seasonality - best months to run campaigns ----------
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
q4 = df.groupby("month")["Converted"].agg(["mean", "count"]).reindex(month_order).dropna().reset_index()
q4["conv_rate_pct"] = (q4["mean"] * 100).round(1)
print("\n--- Q4: Conversion rate by month ---")
print(q4[["month", "conv_rate_pct", "count"]].to_string(index=False))

print("\nAll charts saved.")
