import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load Data and Artifacts
print("Loading saved artifacts...")
X_train = pd.read_parquet("data/processed/X_train.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet")
model = joblib.load("models/lgb_model.pkl")

# Slice the hold-out validation set (last 16 days ~ 1,677,344 rows)
val_len = 1_677_344
X_val = X_train.iloc[-val_len:]
y_val = y_train.iloc[-val_len:]

# 2. Compute Predictions
print("Calculating validation predictions...")
val_preds_log = model.predict(X_val)
val_actual = np.expm1(y_val["log_sales"].values)
val_pred = np.expm1(np.maximum(val_preds_log, 0))

# 3. Create Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot A: Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=axes[0], palette="viridis")
axes[0].set_title("Top 15 Features Determining Sales")
axes[0].set_xlabel("Number of Decision Splits")

# Plot B: Total Daily Sales: Actual vs Predicted
eval_df = pd.DataFrame({
    "dayofyear": X_val["dayofyear"].values,
    "actual": val_actual,
    "pred": val_pred
}).groupby("dayofyear").sum()

axes[1].plot(eval_df.index, eval_df["actual"], label="Actual Sales", marker="o", color="black", linewidth=2)
axes[1].plot(eval_df.index, eval_df["pred"], label="Predicted Sales", marker="s", color="dodgerblue", linestyle="--", linewidth=2)
axes[1].set_title("Total Store Sales (Validation Period)")
axes[1].set_xlabel("Day of Year (August 2017)")
axes[1].set_ylabel("Total Units Sold")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

# Plot C: Error Distribution
errors = val_pred - val_actual
sns.histplot(errors, bins=50, ax=axes[2], color="crimson", kde=True)
axes[2].set_xlim(-20, 20)  # Focus on primary error mass
axes[2].set_title("Prediction Error Spread (Pred - Actual)")
axes[2].set_xlabel("Error in Units")

plt.tight_layout()
plt.savefig("reports/model_diagnostics.png", dpi=200)
print("\nPlot saved successfully to: reports/model_diagnostics.png")
plt.show()