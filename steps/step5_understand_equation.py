import pandas as pd
from pathlib import Path


# ============================================================
# STEP 5 - UNDERSTAND THE MULTIPLE REGRESSION EQUATION
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
print("STEP 5 - UNDERSTAND MULTIPLE LINEAR REGRESSION EQUATION")
print("=" * 70)


# ------------------------------------------------------------
# 3. OUR INPUT FEATURES AND TARGET
# ------------------------------------------------------------

print("\nINPUT FEATURES")
print("=" * 70)

print("X1 = area_sqft")
print("X2 = bedrooms")


print("\nTARGET")
print("=" * 70)

print("y = price_usd")


# ------------------------------------------------------------
# 4. SHOW THE EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION EQUATION")
print("=" * 70)

print(
    "Predicted Price = "
    "b0 + (b1 × Area) + (b2 × Bedrooms)"
)


# ------------------------------------------------------------
# 5. EXPLAIN EACH PART
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT DOES EACH TERM MEAN?")
print("=" * 70)


print(
    "b0 = Intercept"
)

print(
    "b1 = Coefficient for house area"
)

print(
    "b2 = Coefficient for number of bedrooms"
)

print(
    "Area = area_sqft"
)

print(
    "Bedrooms = number of bedrooms"
)


# ------------------------------------------------------------
# 6. COMPARE WITH SIMPLE LINEAR REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIMPLE VS MULTIPLE LINEAR REGRESSION")
print("=" * 70)


print("\nSimple Linear Regression:")

print(
    "Price = b0 + (b1 × Area)"
)


print("\nMultiple Linear Regression:")

print(
    "Price = "
    "b0 + (b1 × Area) + (b2 × Bedrooms)"
)


# ------------------------------------------------------------
# 7. SHOW ONE HOUSE FROM OUR DATASET
# ------------------------------------------------------------

example = df.iloc[0]


example_area = example[
    "area_sqft"
]

example_bedrooms = example[
    "bedrooms"
]

example_price = example[
    "price_usd"
]


print("\n" + "=" * 70)
print("ONE HOUSE FROM OUR DATASET")
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
# 8. ILLUSTRATIVE COEFFICIENTS
# ------------------------------------------------------------

# These numbers are ONLY examples.
#
# They are NOT our trained model coefficients.
#
# We will calculate the real coefficients
# from our dataset in Step 6.


example_intercept = 50000

example_area_coefficient = 180

example_bedroom_coefficient = 15000


print("\n" + "=" * 70)
print("ILLUSTRATIVE EXAMPLE")
print("=" * 70)


print(
    "Suppose:"
)

print(
    f"b0 = ${example_intercept:,.2f}"
)

print(
    f"b1 = ${example_area_coefficient:,.2f}"
)

print(
    f"b2 = ${example_bedroom_coefficient:,.2f}"
)


# ------------------------------------------------------------
# 9. MAKE AN EXAMPLE CALCULATION
# ------------------------------------------------------------

example_prediction = (
    example_intercept
    + example_area_coefficient * example_area
    + example_bedroom_coefficient * example_bedrooms
)


print("\nFormula:")

print(
    "Predicted Price = "
    "b0 + (b1 × Area) + (b2 × Bedrooms)"
)


print("\nCalculation:")

print(
    f"Predicted Price = "
    f"{example_intercept:,.2f}"
    f" + ({example_area_coefficient:,.2f} × {example_area})"
    f" + ({example_bedroom_coefficient:,.2f} × {example_bedrooms})"
)


print(
    f"\nIllustrative Predicted Price = "
    f"${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 10. EXPLAIN COEFFICIENT INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HOW TO INTERPRET THE COEFFICIENTS")
print("=" * 70)


print(
    "Area coefficient:"
)

print(
    "It tells us how much predicted price changes "
    "when area increases by 1 square foot, "
    "while bedrooms stay the same."
)


print("\nBedroom coefficient:")

print(
    "It tells us how much predicted price changes "
    "when bedrooms increase by 1, "
    "while area stays the same."
)


# ------------------------------------------------------------
# 11. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)


print(
    "The coefficients used above were only examples."
)

print(
    "We have NOT trained the model yet."
)

print(
    "In Step 6, we will calculate the actual "
    "intercept and coefficients from our dataset."
)


# ------------------------------------------------------------
# 12. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)