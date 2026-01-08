import pandas as pd
import numpy as np

# ============================================================
# Load data
# ============================================================

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Required columns
FICO_COL = "fico_score"
DEFAULT_COL = "default"

df = df[[FICO_COL, DEFAULT_COL]].sort_values(FICO_COL).reset_index(drop=True)

# ============================================================
# Log-likelihood function for a bucket
# ============================================================

def bucket_log_likelihood(bucket):
    """
    Compute log-likelihood contribution for one bucket.
    """
    n = len(bucket)
    if n == 0:
        return 0

    k = bucket[DEFAULT_COL].sum()
    p = k / n if k > 0 else 1e-6
    p = min(max(p, 1e-6), 1 - 1e-6)

    return k * np.log(p) + (n - k) * np.log(1 - p)

# ============================================================
# Dynamic Programming Quantization
# ============================================================

def optimal_fico_buckets(data, num_buckets):
    """
    Finds optimal FICO bucket boundaries using dynamic programming
    by maximizing log-likelihood.
    """
    scores = data[FICO_COL].values
    N = len(scores)

    dp = np.full((num_buckets + 1, N + 1), -np.inf)
    split = np.zeros((num_buckets + 1, N + 1), dtype=int)

    dp[0][0] = 0

    for b in range(1, num_buckets + 1):
        for i in range(1, N + 1):
            for j in range(b - 1, i):
                ll = dp[b - 1][j] + bucket_log_likelihood(data.iloc[j:i])
                if ll > dp[b][i]:
                    dp[b][i] = ll
                    split[b][i] = j

    # Backtrack to find boundaries
    boundaries = []
    idx = N
    for b in range(num_buckets, 0, -1):
        boundaries.append(scores[split[b][idx]])
        idx = split[b][idx]

    boundaries.reverse()
    return boundaries

# ============================================================
# Build rating map
# ============================================================

def build_rating_map(boundaries):
    """
    Lower rating = better credit
    """
    rating_map = {}
    prev = 0

    for i, boundary in enumerate(boundaries):
        rating_map[i + 1] = (prev, boundary)
        prev = boundary + 1

    rating_map[len(boundaries) + 1] = (prev, 850)
    return rating_map

# ============================================================
# Example execution
# ============================================================

if __name__ == "__main__":

    NUM_BUCKETS = 5

    boundaries = optimal_fico_buckets(df, NUM_BUCKETS)
    rating_map = build_rating_map(boundaries)

    print("Optimal FICO Bucket Boundaries:")
    for i, b in enumerate(boundaries, 1):
        print(f"Bucket {i}: up to FICO {b}")

    print("\nRating Map (lower rating = better credit):")
    for rating, (low, high) in rating_map.items():
        print(f"Rating {rating}: FICO {low} - {high}")
