import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# PART 1: Load data and build gas price estimation model
# ============================================================

df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["t"] = np.arange(len(df))
df["month"] = df["Date"].dt.month

month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
X = pd.concat([df[["t"]], month_dummies], axis=1)
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

START_DATE = df["Date"].iloc[0]

def estimate_price(input_date):
    """
    Estimate natural gas price for a given date.
    """
    date = pd.to_datetime(input_date)
    t = (date.year - START_DATE.year) * 12 + (date.month - START_DATE.month)

    features = {"t": t}
    for m in range(2, 13):
        features[f"month_{m}"] = 1 if date.month == m else 0

    X_input = pd.DataFrame([features])
    return float(model.predict(X_input)[0])

# ============================================================
# PART 2: Storage contract pricing model
# ============================================================

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_storage_volume,
    storage_cost_per_unit
):
    """
    Prices a commodity storage contract.

    Parameters:
        injection_dates (list): Dates when gas is injected
        withdrawal_dates (list): Dates when gas is withdrawn
        injection_rate (float): Max units injected per date
        withdrawal_rate (float): Max units withdrawn per date
        max_storage_volume (float): Storage capacity
        storage_cost_per_unit (float): Cost per unit stored (total)

    Returns:
        float: Net contract value
    """

    storage_volume = 0
    cash_flow = 0

    # Injection phase
    for date in injection_dates:
        price = estimate_price(date)
        injected = min(injection_rate, max_storage_volume - storage_volume)
        storage_volume += injected
        cash_flow -= injected * price

    # Withdrawal phase
    for date in withdrawal_dates:
        price = estimate_price(date)
        withdrawn = min(withdrawal_rate, storage_volume)
        storage_volume -= withdrawn
        cash_flow += withdrawn * price

    # Storage costs
    storage_cost = storage_volume * storage_cost_per_unit
    cash_flow -= storage_cost

    return cash_flow

# ============================================================
# PART 3: Example test cases
# ============================================================

if __name__ == "__main__":
    injection_dates = [
        "2024-01-31",
        "2024-02-29",
        "2024-03-31"
    ]

    withdrawal_dates = [
        "2024-10-31",
        "2024-11-30",
        "2024-12-31"
    ]

    contract_value = price_storage_contract(
        injection_dates=injection_dates,
        withdrawal_dates=withdrawal_dates,
        injection_rate=1000,
        withdrawal_rate=1000,
        max_storage_volume=3000,
        storage_cost_per_unit=0.05
    )

    print(f"Storage contract value: {contract_value:.2f}")
