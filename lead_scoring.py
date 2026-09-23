"""
Lead scoring and contact-policy optimisation for a bank term-deposit campaign.

Data: UCI Bank Marketing (bank.csv, 4,521 rows, ';' separated).
Run:  python lead_scoring.py path/to/bank.csv

Business question: if the bank can only afford to call some customers, who should it
call, and how many, to earn the most from the campaign?
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Business assumptions (ASSUMPTIONS, not facts from the data: change and justify) ----
COST_PER_CALL = 5.0        # cost of one call (agent time, telephony)
VALUE_PER_SIGNUP = 100.0   # value to the bank of one term-deposit sign-up
SEED = 42

path = sys.argv[1] if len(sys.argv) > 1 else "bank.csv"
df = pd.read_csv(path, sep=";")
y = (df["y"] == "yes").astype(int)

# 'duration' is call length, which is only known AFTER the call, so using it to decide
# who to call is data leakage. It is dropped on purpose.
X = df.drop(columns=["y", "duration"]).copy()
X["previously_contacted"] = (X["pdays"] != -1).astype(int)   # pdays == -1 means never contacted
X["pdays"] = X["pdays"].replace(-1, 0)

num = ["age", "balance", "day", "campaign", "pdays", "previous", "previously_contacted"]
cat = [c for c in X.columns if c not in num]

def make(model, scale):
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("model", model)])

models = {
    "Logistic regression (baseline)": make(LogisticRegression(max_iter=2000, class_weight=None), True),
    "Gradient boosting": make(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=200, random_state=SEED), False),
}

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

print(f"Conversion rate: train {y_tr.mean():.1%}, test {y_te.mean():.1%}\n")
rows, fitted = [], {}
for name, m in models.items():
    oof = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    fitted[name] = (m, oof, p)
    rows.append([name, roc_auc_score(y_tr, oof), roc_auc_score(y_te, p),
                 average_precision_score(y_te, p), brier_score_loss(y_te, p)])
res = pd.DataFrame(rows, columns=["model", "cv_auc", "test_auc", "test_pr_auc", "test_brier"])
print(res.round(3).to_string(index=False), "\n")

best = res.sort_values("cv_auc").iloc[-1]["model"]
model, oof, p_te = fitted[best]
print(f"Model used for the policy: {best}\n")

# ---- Gains / lift by decile on the test set ----
t = pd.DataFrame({"p": p_te, "y": y_te.values}).sort_values("p", ascending=False).reset_index(drop=True)
t["decile"] = pd.qcut(t.index, 10, labels=range(1, 11))
g = t.groupby("decile", observed=True).agg(customers=("y", "size"), signups=("y", "sum"))
g["conv_rate"] = g["signups"] / g["customers"]
g["lift"] = g["conv_rate"] / t["y"].mean()
g["cum_share_of_signups"] = g["signups"].cumsum() / t["y"].sum()
print(g.round(3).to_string(), "\n")

# ---- Choose the contact policy WITHOUT touching the test set ----
# Pick the score threshold on out-of-fold training predictions, then evaluate on test.
def profit(p_true_y, contacted):
    return (p_true_y * contacted).sum() * VALUE_PER_SIGNUP - contacted.sum() * COST_PER_CALL

order = np.argsort(-oof)
y_sorted = y_tr.values[order]
cum_profit = np.cumsum(y_sorted * VALUE_PER_SIGNUP - COST_PER_CALL)
k = int(np.argmax(cum_profit)) + 1
threshold = oof[order][k - 1]
print(f"Policy chosen on training data: call customers with score >= {threshold:.3f} "
      f"(top {k / len(oof):.0%} of the list)")

call = p_te >= threshold
policy = profit(y_te.values, call.astype(int))
call_all = profit(y_te.values, np.ones(len(y_te), dtype=int))
print(f"Test set profit  - call everyone: {call_all:,.0f} | model policy: {policy:,.0f} "
      f"| calls made: {call.sum()} of {len(call)} ({call.mean():.0%}) "
      f"| sign-ups captured: {y_te.values[call].sum() / y_te.sum():.0%}\n")

# ---- Sensitivity: does the policy survive different cost assumptions? ----
print("Sensitivity (test profit, model policy vs call everyone):")
for c in (2, 5, 10, 20):
    o = np.argsort(-oof)
    cp = np.cumsum(y_tr.values[o] * VALUE_PER_SIGNUP - c)
    th = oof[o][int(np.argmax(cp))]
    m_ = (p_te >= th).astype(int)
    a = (y_te.values * m_).sum() * VALUE_PER_SIGNUP - m_.sum() * c
    b = y_te.sum() * VALUE_PER_SIGNUP - len(y_te) * c
    print(f"  cost/call {c:>3}: model {a:>9,.0f} | everyone {b:>9,.0f}")

# ---- What drives the score? ----
imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
imp = pd.Series(imp.importances_mean, index=X_te.columns).sort_values(ascending=False)
print("\nTop drivers (drop in test AUC when the feature is shuffled):")
print(imp.head(6).round(4).to_string())

# ---- Charts ----
share_called = np.arange(1, len(t) + 1) / len(t)

# Chart 1: cumulative gains
plt.figure(figsize=(6.5, 4.5))
plt.plot(share_called, t["y"].cumsum() / t["y"].sum(), lw=2, label="Ranked by model score")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random calling")
plt.title("Calling top-scored customers first beats random calling")
plt.xlabel("Share of customers called"); plt.ylabel("Share of sign-ups captured")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("chart1_cumulative_gains.png", dpi=150); plt.close()

# Chart 2: does the model beat "call everyone" as call cost changes?
costs = np.arange(1, 31)
model_p, all_p = [], []
for c in costs:
    o = np.argsort(-oof)
    th = oof[o][int(np.argmax(np.cumsum(y_tr.values[o] * VALUE_PER_SIGNUP - c)))]
    m_ = (p_te >= th).astype(int)
    model_p.append((y_te.values * m_).sum() * VALUE_PER_SIGNUP - m_.sum() * c)
    all_p.append(y_te.sum() * VALUE_PER_SIGNUP - len(y_te) * c)
plt.figure(figsize=(6.5, 4.5))
plt.plot(costs, model_p, lw=2, label="Call by model score")
plt.plot(costs, all_p, lw=2, label="Call everyone")
plt.axhline(0, color="grey", lw=.8)
plt.title(f"Profit vs cost per call (sign-up worth {VALUE_PER_SIGNUP:.0f})")
plt.xlabel("Cost per call"); plt.ylabel("Profit on test set")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("chart2_profit_vs_call_cost.png", dpi=150); plt.close()

# Chart 3: drivers
top = imp.head(8)[::-1]
plt.figure(figsize=(6.5, 4.5))
plt.barh(top.index, top.values)
plt.title("What drives the score (permutation importance)")
plt.xlabel("Drop in test AUC when the feature is shuffled")
plt.tight_layout(); plt.savefig("chart3_drivers.png", dpi=150); plt.close()
print("\nSaved chart1_cumulative_gains.png, chart2_profit_vs_call_cost.png, chart3_drivers.png")
