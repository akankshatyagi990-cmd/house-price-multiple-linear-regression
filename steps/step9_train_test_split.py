import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 9 - TRAIN / TEST SPLIT
# FOR MULTIPLE LINEAR REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_multiple_regression.csv"
)

OUTPUTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

TRAIN_OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step9_train_data.csv"
)

TEST_OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step9_test_data.csv"
)


# ------------------------------------------------------------
# 2. CREATE OUTPUTS FOLDER
# ------------------------------------------------------------

OUTPUTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 9 - TRAIN / TEST SPLIT")
print("=" * 70)


# ------------------------------------------------------------
# 4. SELECT INPUT FEATURES
# ------------------------------------------------------------

X = df[
    [
        "area_sqft",
        "bedrooms",
    ]
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# 5. SELECT TARGET
# ------------------------------------------------------------

y = df[
    "price_usd"
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# 6. CHECK TOTAL NUMBER OF ROWS
# ------------------------------------------------------------

total_rows = len(df)


print("\nTOTAL DATASET SIZE")
print("=" * 70)

print(
    f"Total rows = {total_rows}"
)


# ------------------------------------------------------------
# 7. CREATE RANDOM INDEXES
# ------------------------------------------------------------

# Random seed ensures the same split
# every time we run the script.

np.random.seed(
    42
)


indices = np.arange(
    total_rows
)


np.random.shuffle(
    indices
)


# ------------------------------------------------------------
# 8. DEFINE TRAINING SIZE
# ------------------------------------------------------------

train_ratio = 0.80


train_size = int(
    total_rows
    * train_ratio
)


# ------------------------------------------------------------
# 9. SPLIT INDEXES
# ------------------------------------------------------------

train_indices = (
    indices[:train_size]
)

test_indices = (
    indices[train_size:]
)


# ------------------------------------------------------------
# 10. CREATE TRAINING DATA
# ------------------------------------------------------------

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


# ------------------------------------------------------------
# 11. CREATE TEST DATA
# ------------------------------------------------------------

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]


# ------------------------------------------------------------
# 12. DISPLAY SPLIT SIZE
# ------------------------------------------------------------

print("\nTRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training rows = {len(X_train)}"
)

print(
    f"Testing rows = {len(X_test)}"
)

print(
    f"Training percentage = "
    f"{len(X_train) / total_rows * 100:.0f}%"
)

print(
    f"Testing percentage = "
    f"{len(X_test) / total_rows * 100:.0f}%"
)


# ------------------------------------------------------------
# 13. CREATE TRAINING DESIGN MATRIX
# ------------------------------------------------------------

ones_train = np.ones(
    (
        X_train.shape[0],
        1,
    )
)


X_train_design = np.hstack(
    (
        ones_train,
        X_train,
    )
)


# ------------------------------------------------------------
# 14. TRAIN MODEL USING TRAINING DATA ONLY
# ------------------------------------------------------------

beta = np.linalg.lstsq(
    X_train_design,
    y_train,
    rcond=None,
)[0]


intercept = beta[0]

area_coefficient = beta[1]

bedroom_coefficient = beta[2]


# ------------------------------------------------------------
# 15. DISPLAY TRAINED MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINED USING TRAINING DATA ONLY")
print("=" * 70)


print(
    f"Intercept = "
    f"${intercept:,.2f}"
)

print(
    f"Area coefficient = "
    f"${area_coefficient:,.2f}"
)

print(
    f"Bedroom coefficient = "
    f"${bedroom_coefficient:,.2f}"
)


print("\nRegression Equation:")


print(
    "Predicted Price ="
)

print(
    f"{intercept:,.2f}"
)

print(
    f"+ ({area_coefficient:,.2f} × Area)"
)

print(
    f"+ ({bedroom_coefficient:,.2f} × Bedrooms)"
)


# ------------------------------------------------------------
# 16. CREATE TEST DESIGN MATRIX
# ------------------------------------------------------------

ones_test = np.ones(
    (
        X_test.shape[0],
        1,
    )
)


X_test_design = np.hstack(
    (
        ones_test,
        X_test,
    )
)


# ------------------------------------------------------------
# 17. MAKE PREDICTIONS ON TEST DATA
# ------------------------------------------------------------

y_test_predicted = (
    X_test_design
    @ beta
)


# ------------------------------------------------------------
# 18. CREATE TRAINING DATAFRAME
# ------------------------------------------------------------

train_data = pd.DataFrame(
    {
        "area_sqft": X_train[:, 0],
        "bedrooms": X_train[:, 1],
        "actual_price_usd": y_train,
    }
)


# ------------------------------------------------------------
# 19. CREATE TEST DATAFRAME
# ------------------------------------------------------------

test_data = pd.DataFrame(
    {
        "area_sqft": X_test[:, 0],
        "bedrooms": X_test[:, 1],
        "actual_price_usd": y_test,
        "predicted_price_usd": y_test_predicted,
    }
)


test_data[
    "predicted_price_usd"
] = (
    test_data[
        "predicted_price_usd"
    ].round(2)
)


# ------------------------------------------------------------
# 20. DISPLAY FIRST 10 TRAINING ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TRAINING ROWS")
print("=" * 70)


print(
    train_data.head(10)
)


# ------------------------------------------------------------
# 21. DISPLAY FIRST 10 TEST PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TEST PREDICTIONS")
print("=" * 70)


print(
    test_data.head(10)
)


# ------------------------------------------------------------
# 22. SAVE TRAINING DATA
# ------------------------------------------------------------

train_data.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 23. SAVE TEST DATA
# ------------------------------------------------------------

test_data.to_csv(
    TEST_OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 24. SHOW SAVED FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)


print(
    TRAIN_OUTPUT_FILE
)

print(
    TEST_OUTPUT_FILE
)


# ------------------------------------------------------------
# 25. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)