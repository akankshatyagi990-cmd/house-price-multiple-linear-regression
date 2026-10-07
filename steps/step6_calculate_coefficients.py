import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 6 - CALCULATE MULTIPLE REGRESSION COEFFICIENTS
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
print("STEP 6 - CALCULATE MULTIPLE REGRESSION COEFFICIENTS")
print("=" * 70)


# ------------------------------------------------------------
# 3. SELECT INPUT FEATURES
# ------------------------------------------------------------

# X has TWO features:
#
# X1 = area_sqft
# X2 = bedrooms

X = df[
    [
        "area_sqft",
        "bedrooms",
    ]
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# 4. SELECT TARGET
# ------------------------------------------------------------

y = df[
    "price_usd"
].to_numpy(
    dtype=float
)


print("\nINPUT FEATURES")

print("X1 = area_sqft")
print("X2 = bedrooms")

print("\nTARGET")

print("y = price_usd")


# ------------------------------------------------------------
# 5. CHECK X SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ORIGINAL X MATRIX")
print("=" * 70)

print(
    X[:5]
)

print(
    f"\nX shape = {X.shape}"
)


# ------------------------------------------------------------
# 6. ADD COLUMN OF 1s FOR INTERCEPT
# ------------------------------------------------------------

# Our equation is:
#
# Price =
# b0
# + b1 * Area
# + b2 * Bedrooms
#
# To calculate b0 together with b1 and b2,
# we add a column of 1s.


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


print("\n" + "=" * 70)
print("DESIGN MATRIX")
print("=" * 70)

print(
    X_design[:5]
)

print(
    f"\nDesign matrix shape = "
    f"{X_design.shape}"
)


# ------------------------------------------------------------
# 7. CALCULATE COEFFICIENTS USING LEAST SQUARES
# ------------------------------------------------------------

# beta contains:
#
# beta[0] = intercept
# beta[1] = area coefficient
# beta[2] = bedroom coefficient


beta = np.linalg.lstsq(
    X_design,
    y,
    rcond=None,
)[0]


# ------------------------------------------------------------
# 8. SEPARATE COEFFICIENTS
# ------------------------------------------------------------

intercept = beta[0]

area_coefficient = beta[1]

bedroom_coefficient = beta[2]


# ------------------------------------------------------------
# 9. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATED COEFFICIENTS")
print("=" * 70)


print(
    f"Intercept (b0) = "
    f"${intercept:,.2f}"
)

print(
    f"Area coefficient (b1) = "
    f"${area_coefficient:,.2f}"
)

print(
    f"Bedroom coefficient (b2) = "
    f"${bedroom_coefficient:,.2f}"
)


# ------------------------------------------------------------
# 10. DISPLAY FINAL EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MULTIPLE REGRESSION EQUATION")
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
# 11. EXPLAIN AREA COEFFICIENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AREA COEFFICIENT INTERPRETATION")
print("=" * 70)


print(
    f"If house area increases by 1 square foot, "
    f"the predicted price changes by approximately "
    f"${area_coefficient:,.2f},"
)

print(
    "while keeping the number of bedrooms constant."
)


# ------------------------------------------------------------
# 12. EXPLAIN BEDROOM COEFFICIENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEDROOM COEFFICIENT INTERPRETATION")
print("=" * 70)


print(
    f"If the number of bedrooms increases by 1, "
    f"the predicted price changes by approximately "
    f"${bedroom_coefficient:,.2f},"
)

print(
    "while keeping house area constant."
)


# ------------------------------------------------------------
# 13. EXPLAIN INTERCEPT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERCEPT INTERPRETATION")
print("=" * 70)


print(
    f"Intercept = ${intercept:,.2f}"
)

print(
    "This represents the predicted price when:"
)

print(
    "Area = 0"
)

print(
    "Bedrooms = 0"
)

print(
    "That is not a realistic house, so the intercept "
    "mainly helps position the regression model correctly."
)


# ------------------------------------------------------------
# 14. MAKE ONE EXAMPLE PREDICTION
# ------------------------------------------------------------

example_area = 2000

example_bedrooms = 3


example_prediction = (
    intercept
    + area_coefficient * example_area
    + bedroom_coefficient * example_bedrooms
)


print("\n" + "=" * 70)
print("EXAMPLE PREDICTION")
print("=" * 70)


print(
    f"Area = {example_area} sq ft"
)

print(
    f"Bedrooms = {example_bedrooms}"
)


print("\nCalculation:")


print(
    f"Predicted Price = "
    f"{intercept:,.2f}"
    f" + ({area_coefficient:,.2f} × {example_area})"
    f" + ({bedroom_coefficient:,.2f} × {example_bedrooms})"
)


print(
    f"\nPredicted Price = "
    f"${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)
