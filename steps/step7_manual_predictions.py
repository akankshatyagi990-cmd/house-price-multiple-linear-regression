import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 7 - MAKE MULTIPLE REGRESSION PREDICTIONS MANUALLY
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

OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step7_manual_predictions.csv"
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
print("STEP 7 - MAKE MULTIPLE REGRESSION PREDICTIONS")
print("=" * 70)


# ------------------------------------------------------------
# 4. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

X = df[
    [
        "area_sqft",
        "bedrooms",
    ]
].to_numpy(
    dtype=float
)


y = df[
    "price_usd"
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# 5. CREATE DESIGN MATRIX
# ------------------------------------------------------------

ones = np.ones(
    (
        X.shape[0],
        1,
    )
)


X_design = np.hstack(
    (
        ones,
        X,
    )
)


# ------------------------------------------------------------
# 6. CALCULATE COEFFICIENTS
# ------------------------------------------------------------

beta = np.linalg.lstsq(
    X_design,
    y,
    rcond=None,
)[0]


intercept = beta[0]

area_coefficient = beta[1]

bedroom_coefficient = beta[2]


# ------------------------------------------------------------
# 7. DISPLAY EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION EQUATION")
print("=" * 70)


print(
    "Predicted Price = "
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
# 8. CALCULATE PREDICTIONS FOR ALL HOUSES
# ------------------------------------------------------------

predicted_prices = (
    intercept
    + area_coefficient * df["area_sqft"]
    + bedroom_coefficient * df["bedrooms"]
)


# ------------------------------------------------------------
# 9. CREATE RESULTS DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "house_id": df["house_id"],
        "area_sqft": df["area_sqft"],
        "bedrooms": df["bedrooms"],
        "actual_price_usd": df["price_usd"],
        "predicted_price_usd": predicted_prices,
    }
)


# ------------------------------------------------------------
# 10. ROUND PREDICTIONS
# ------------------------------------------------------------

results[
    "predicted_price_usd"
] = (
    results[
        "predicted_price_usd"
    ].round(2)
)


# ------------------------------------------------------------
# 11. DISPLAY FIRST 10 PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 PREDICTIONS")
print("=" * 70)


print(
    results.head(10)
)


# ------------------------------------------------------------
# 12. EXPLAIN ONE HOUSE
# ------------------------------------------------------------

example = results.iloc[0]


example_area = example[
    "area_sqft"
]

example_bedrooms = example[
    "bedrooms"
]

example_actual = example[
    "actual_price_usd"
]


example_prediction = (
    intercept
    + area_coefficient * example_area
    + bedroom_coefficient * example_bedrooms
)


print("\n" + "=" * 70)
print("ONE PREDICTION EXPLAINED")
print("=" * 70)


print(
    f"House ID = {example['house_id']}"
)

print(
    f"Area = {example_area:,.0f} sq ft"
)

print(
    f"Bedrooms = {example_bedrooms:,.0f}"
)

print(
    f"Actual Price = "
    f"${example_actual:,.2f}"
)


print("\nFormula:")

print(
    "Predicted Price = "
    "Intercept "
    "+ Area Coefficient × Area "
    "+ Bedroom Coefficient × Bedrooms"
)


print("\nCalculation:")

print(
    f"Predicted Price = "
    f"{intercept:,.2f}"
    f" + ({area_coefficient:,.2f} × {example_area:,.0f})"
    f" + ({bedroom_coefficient:,.2f} × {example_bedrooms:,.0f})"
)


print(
    f"\nPredicted Price = "
    f"${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 13. TRY CUSTOM HOUSE EXAMPLES
# ------------------------------------------------------------

custom_houses = [
    [1200, 2],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 5],
]


print("\n" + "=" * 70)
print("CUSTOM HOUSE PREDICTIONS")
print("=" * 70)


for area, bedrooms in custom_houses:

    prediction = (
        intercept
        + area_coefficient * area
        + bedroom_coefficient * bedrooms
    )

    print(
        f"{area:>4} sq ft | "
        f"{bedrooms} bedrooms "
        f"-> ${prediction:,.2f}"
    )


# ------------------------------------------------------------
# 14. SAVE PREDICTIONS
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("PREDICTIONS FILE SAVED")
print("=" * 70)


print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)