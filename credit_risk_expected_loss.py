import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ============================================================
# PART 1: Load the dataset
# ============================================================

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Target column: whether borrower defaulted
TARGET = "default"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================================================
# PART 2: Train PD (Probability of Default) model
# ============================================================

# Pipeline with scaling + logistic regression
pd_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pd_model.fit(X_train, y_train)

# Model evaluation (good practice)
pd_test = pd_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_test)
print(f"Model ROC-AUC: {auc:.3f}")

# ============================================================
# PART 3: Expected Loss calculation
# ============================================================

RECOVERY_RATE = 0.10  # given
LGD = 1 - RECOVERY_RATE

def expected_loss(loan_features, exposure):
    """
    Calculate expected loss for a loan.

    Parameters:
        loan_features (dict): borrower features
        exposure (float): loan amount (EAD)

    Returns:
        float: expected loss
    """
    features_df = pd.DataFrame([loan_features])
    pd_estimate = pd_model.predict_proba(features_df)[0][1]
    return pd_estimate * LGD * exposure

# ============================================================
# PART 4: Example test case
# ============================================================

if __name__ == "__main__":

    sample_loan = X.iloc[0].to_dict()
    exposure = sample_loan.get("loan_amount", 10000)

    el = expected_loss(sample_loan, exposure)

    print(f"Estimated Probability of Default: {pd_model.predict_proba(pd.DataFrame([sample_loan]))[0][1]:.3f}")
    print(f"Expected Loss on Loan: {el:.2f}")
