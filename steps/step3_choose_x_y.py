import pandas as pd
from pathlib import Path


# ============================================================
# STEP 3 - CHOOSE MULTIPLE X FEATURES AND y
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
print("STEP 3 - CHOOSE MULTIPLE X FEATURES AND y")
print("=" * 70)


# ------------------------------------------------------------
# 3. CHOOSE INPUT FEATURES X
# ------------------------------------------------------------

X = df[
    [
        "area_sqft",
        "bedrooms",
    ]
]


# ------------------------------------------------------------
# 4. CHOOSE TARGET y
# ------------------------------------------------------------

y = df[
    "price_usd"
]


# ------------------------------------------------------------
# 5. DISPLAY OUR VARIABLES
# ------------------------------------------------------------

print("\nINPUT FEATURES")
print("=" * 70)

print("X1 = area_sqft")
print("X2 = bedrooms")


print("\nTARGET")
print("=" * 70)

print("y = price_usd")


# ------------------------------------------------------------
# 6. DISPLAY FIRST 10 X VALUES
# ------------------------------------------------------------

print("\nFIRST 10 X VALUES")
print("=" * 70)

print(
    X.head(10)
)


# ------------------------------------------------------------
# 7. DISPLAY FIRST 10 y VALUES
# ------------------------------------------------------------

print("\nFIRST 10 y VALUES")
print("=" * 70)

print(
    y.head(10)
)


# ------------------------------------------------------------
# 8. DISPLAY X AND y TOGETHER
# ------------------------------------------------------------

model_data = pd.DataFrame(
    {
        "area_sqft": X["area_sqft"],
        "bedrooms": X["bedrooms"],
        "price_usd": y,
    }
)


print("\n" + "=" * 70)
print("FIRST 10 MODEL EXAMPLES")
print("=" * 70)

print(
    model_data.head(10)
)


# ------------------------------------------------------------
# 9. CHECK X SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SHAPE OF X")
print("=" * 70)

print(
    X.shape
)


print(
    f"Rows in X = {X.shape[0]}"
)

print(
    f"Features in X = {X.shape[1]}"
)


# ------------------------------------------------------------
# 10. CHECK y SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SHAPE OF y")
print("=" * 70)

print(
    y.shape
)


print(
    f"Values in y = {len(y)}"
)


# ------------------------------------------------------------
# 11. SHOW ONE PRACTICAL EXAMPLE
# ------------------------------------------------------------

example_area = X.iloc[0]["area_sqft"]

example_bedrooms = X.iloc[0]["bedrooms"]

example_price = y.iloc[0]


print("\n" + "=" * 70)
print("ONE EXAMPLE")
print("=" * 70)

print(
    f"Area = {example_area} sq ft"
)

print(
    f"Bedrooms = {example_bedrooms}"
)

print(
    f"Actual Price = ${example_price:,.2f}"
)


# ------------------------------------------------------------
# 12. EXPLAIN MODEL INPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL IDEA")
print("=" * 70)

print(
    "We will use BOTH area and bedrooms "
    "to predict house price."
)


print(
    "\n(area_sqft + bedrooms) -> price_usd"
)


# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)