import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

HOLD_COST = 1500
SEED = 42

# 1. Load and prep
cars = pd.read_csv("used_car_sales.csv")
lots = pd.read_csv("auction_lots.csv")
print("Columns:", list(cars.columns))
print("Rows:", len(cars))
print(cars.isna().sum())
print(cars.describe())

for df in (cars, lots):
    df["Accident"] = (df["Accident"].astype(str).str.strip().str.lower() == "yes").astype(int)

# 2. 70/30 split
train, test = train_test_split(cars, test_size=0.30, random_state=SEED)
print("\nTrain:", len(train), "Test:", len(test))

# 3. Fit models
formula_v1 = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident + Num_Owners"
formula_v2 = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident"
v1 = smf.ols(formula_v1, data=train).fit()
v2 = smf.ols(formula_v2, data=train).fit()
print(v1.summary())
print("\nAdj R2 (all 5 features):", round(v1.rsquared_adj, 4))
print("Adj R2 (without Num_Owners):", round(v2.rsquared_adj, 4))

final = v2
print("\n===== FINAL MODEL (without Num_Owners) =====")
print(final.summary())

# 4. Evaluate on test cars
baseline_pred = np.full(len(test), train["Sale_Price"].mean())
model_pred = final.predict(test)

def report(name, pred):
    r2 = r2_score(test["Sale_Price"], pred)
    rmse = np.sqrt(mean_squared_error(test["Sale_Price"], pred))
    print(f"{name}: R2 = {r2:.3f}, typical error (RMSE) = ${rmse:,.0f}")

print("\n--- TEST SET ---")
report("Baseline", baseline_pred)
report("Model   ", model_pred)

plt.figure(figsize=(6, 5))
plt.scatter(test["Sale_Price"], model_pred, alpha=0.7, label="Model")
lim = [test["Sale_Price"].min(), test["Sale_Price"].max()]
plt.plot(lim, lim, "k--", label="Perfect prediction")
plt.axhline(train["Sale_Price"].mean(), color="gray", label="Baseline (average)")
plt.xlabel("Actual sale price ($)")
plt.ylabel("Predicted sale price ($)")
plt.legend()
plt.savefig("pred_vs_actual.png", dpi=200, bbox_inches="tight")

# 5. Decision rule
def decide(car, auction_price, model=final):
    """car: dict with Mileage_K, Age_Years, Horsepower, Accident (0/1). """
    predicted = model.predict(pd.DataFrame([car]))[0]
    return "buy" if auction_price + HOLD_COST < predicted else "pass"

feature_cols = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]
avg = train["Sale_Price"].mean()
lots["profit"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - HOLD_COST
lots["model_decision"] = [
    decide(row[feature_cols].to_dict(), row["Auction_Price"]) for _, row in lots.iterrows()
]
lots["baseline_decision"] = np.where(lots["Auction_Price"] + HOLD_COST < avg, "buy", "pass")

print("\n--- 40 AUCTION LOTS ---")
for name, col in [("Baseline", "baseline_decision"), ("Model", "model_decision")]:
    bought = lots[lots[col] == "buy"]
    print(f"{name}: bought {len(bought)}, losers {(bought['profit'] < 0).sum()}, "
          f"total profit ${bought['profit'].sum():,.0f}")

lots.to_csv("auction_decisions.csv", index=False)
example = {"Mileage_K": 70, "Age_Years": 5, "Horsepower": 200, "Accident": 0}
print("\nExample car, auction price $18,000:", decide(example, 18000))

