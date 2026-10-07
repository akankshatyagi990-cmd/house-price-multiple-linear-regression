import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 4 - VISUALIZE MULTIPLE FEATURES
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

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

AREA_PLOT = (
    PLOTS_FOLDER
    / "step4_area_vs_price.png"
)

BEDROOM_PLOT = (
    PLOTS_FOLDER
    / "step4_bedrooms_vs_price.png"
)


# ------------------------------------------------------------
# 2. CREATE PLOTS FOLDER
# ------------------------------------------------------------

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)


print("\n" + "=" * 70)
print("STEP 4 - VISUALIZE MULTIPLE FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# 4. SELECT VARIABLES
# ------------------------------------------------------------

area = df["area_sqft"]

bedrooms = df["bedrooms"]

price = df["price_usd"]


# ------------------------------------------------------------
# 5. AREA VS PRICE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    area,
    price,
    alpha=0.7
)

plt.xlabel(
    "House Area (Square Feet)"
)

plt.ylabel(
    "House Price (USD)"
)

plt.title(
    "House Area vs House Price"
)

plt.grid(
    alpha=0.3
)

plt.savefig(
    AREA_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 6. BEDROOMS VS PRICE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    bedrooms,
    price,
    alpha=0.7
)

plt.xlabel(
    "Number of Bedrooms"
)

plt.ylabel(
    "House Price (USD)"
)

plt.title(
    "Bedrooms vs House Price"
)

plt.grid(
    alpha=0.3
)

plt.savefig(
    BEDROOM_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 7. SHOW OUTPUT LOCATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PLOTS CREATED")
print("=" * 70)

print(
    AREA_PLOT
)

print(
    BEDROOM_PLOT
)


# ------------------------------------------------------------
# 8. EXPLANATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT TO OBSERVE")
print("=" * 70)

print(
    "Plot 1 shows how house area relates to house price."
)

print(
    "Plot 2 shows how number of bedrooms relates to house price."
)

print(
    "Multiple linear regression will use both features together."
)


# ------------------------------------------------------------
# 9. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)