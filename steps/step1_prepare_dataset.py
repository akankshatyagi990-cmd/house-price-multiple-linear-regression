import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1 - PREPARE DATASET FOR MULTIPLE LINEAR REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_linear_regression_practice.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_multiple_regression.csv"
)


# ------------------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print("\n" + "=" * 70)
print("STEP 1 - PREPARE DATASET")
print("=" * 70)


# ------------------------------------------------------------
# 3. SHOW FIRST 5 ROWS
# ------------------------------------------------------------

print("\nFIRST 5 ROWS")
print("=" * 70)

print(
    df.head()
)


# ------------------------------------------------------------
# 4. SHOW AVAILABLE COLUMNS
# ------------------------------------------------------------

print("\nAVAILABLE COLUMNS")
print("=" * 70)

for column in df.columns:
    print(column)


# ------------------------------------------------------------
# 5. CHOOSE ONLY THE COLUMNS WE NEED
# ------------------------------------------------------------

prepared_df = df[
    [
        "house_id",
        "area_sqft",
        "bedrooms",
        "price_usd",
    ]
].copy()


# ------------------------------------------------------------
# 6. SHOW PREPARED DATA
# ------------------------------------------------------------

print("\nPREPARED DATASET")
print("=" * 70)

print(
    prepared_df.head(10)
)


# ------------------------------------------------------------
# 7. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nDATA TYPES")
print("=" * 70)

print(
    prepared_df.dtypes
)


# ------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("=" * 70)

print(
    prepared_df.isnull().sum()
)


# ------------------------------------------------------------
# 9. CHECK DUPLICATES
# ------------------------------------------------------------

duplicates = (
    prepared_df
    .duplicated()
    .sum()
)


print("\nDUPLICATE ROWS")
print("=" * 70)

print(
    f"Duplicate rows = {duplicates}"
)


# ------------------------------------------------------------
# 10. SHOW MODEL VARIABLES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION VARIABLES")
print("=" * 70)

print("X1 = area_sqft")
print("X2 = bedrooms")
print("y  = price_usd")


# ------------------------------------------------------------
# 11. SAVE PREPARED DATASET
# ------------------------------------------------------------

prepared_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 12. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 70)
