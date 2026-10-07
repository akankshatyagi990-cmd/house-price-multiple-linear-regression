import pandas as pd
from pathlib import Path


# ============================================================
# STEP 2 - EXPLORE THE MULTIPLE REGRESSION DATASET
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_multiple_regression.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 2 - EXPLORE DATASET")
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
# 4. SHOW LAST 5 ROWS
# ------------------------------------------------------------

print("\nLAST 5 ROWS")
print("=" * 70)

print(
    df.tail()
)


# ------------------------------------------------------------
# 5. CHECK DATASET SIZE
# ------------------------------------------------------------

rows, columns = df.shape


print("\nDATASET SIZE")
print("=" * 70)

print(
    f"Number of rows = {rows}"
)

print(
    f"Number of columns = {columns}"
)


# ------------------------------------------------------------
# 6. SHOW COLUMN NAMES
# ------------------------------------------------------------

print("\nCOLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(column)


# ------------------------------------------------------------
# 7. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nDATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("=" * 70)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# 9. CHECK DUPLICATES
# ------------------------------------------------------------

duplicates = (
    df.duplicated().sum()
)


print("\nDUPLICATE ROWS")
print("=" * 70)

print(
    f"Duplicate rows = {duplicates}"
)


# ------------------------------------------------------------
# 10. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    df.describe().round(2)
)


# ------------------------------------------------------------
# 11. CHECK AREA SUMMARY
# ------------------------------------------------------------

print("\nAREA SUMMARY")
print("=" * 70)

print(
    f"Minimum area = "
    f"{df['area_sqft'].min()} sq ft"
)

print(
    f"Maximum area = "
    f"{df['area_sqft'].max()} sq ft"
)

print(
    f"Average area = "
    f"{df['area_sqft'].mean():.2f} sq ft"
)


# ------------------------------------------------------------
# 12. CHECK BEDROOM SUMMARY
# ------------------------------------------------------------

print("\nBEDROOM SUMMARY")
print("=" * 70)

print(
    df["bedrooms"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 13. CHECK PRICE SUMMARY
# ------------------------------------------------------------

print("\nPRICE SUMMARY")
print("=" * 70)

print(
    f"Minimum price = "
    f"${df['price_usd'].min():,.2f}"
)

print(
    f"Maximum price = "
    f"${df['price_usd'].max():,.2f}"
)

print(
    f"Average price = "
    f"${df['price_usd'].mean():,.2f}"
)


# ------------------------------------------------------------
# 14. SHOW RELEVANT VARIABLES TOGETHER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 MODEL ROWS")
print("=" * 70)

print(
    df[
        [
            "area_sqft",
            "bedrooms",
            "price_usd",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)