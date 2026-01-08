import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ============================================================
# PART 1: Load and prepare borrower data
# ============================================================

# CSV should contain borrower features + 'default' column (0 or 1)
# Example columns:
# income, total_loans, credit_score, loan_amount, default

df = pd.read_csv("borrower_data.csv")

# Separate features and target
X = df.drop(columns=["default"])
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ============================================================
# PART 2: Train Probability of Default (PD) model
# ============================================================

pd_model = LogisticRegression(max_iter=1000)
pd_model.fit(X_train, y_train)

# Evaluate model (optional but good practice)
y_pred_prob = pd_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"Model ROC-AUC: {auc:.3f}")

# ============================================================
# PART 3: Expected Loss calculation
# ============================================================

RECOVERY_RATE = 0.10

def expected_loss(loan_features, exposure):
    """
    Calculate expected loss for a loan.

    Parameters:
        loan_features (dict): Borrower features matching training columns
        exposure (float): Loan amount (EAD)

    Returns:
        float: Expected loss
    """
    features_df = pd.DataFrame([loan_features])
    pd_estimate = pd_model.predict_proba(features_df)[0][1]

    loss_given_default = 1 - RECOVERY_RATE
    el = pd_estimate * loss_given_default * exposure

    return el

# ============================================================
# PART 4: Example usage
# ============================================================

if __name__ == "__main__":
    sample_borrower = {
        "income": 60000,
        "total_loans": 15000,
        "credit_score": 680,
        "loan_amount": 10000
    }

    exposure = 10000  # Loan amount

    el = expected_loss(sample_borrower, exposure)
    print(f"Expected Loss for the loan: {el:.2f}")
