import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 8 - CALCULATE RESIDUALS
# FOR MULTIPLE LINEAR REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_manual_predictions.csv"
)

OUTPUTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
)

PLOTS_FOLDER = (
    PROJECT_ROOT
    / "plots"
)

OUTPUT_FILE = (
    OUTPUTS_FOLDER
    / "step8_residuals.csv"
)

OUTPUT_PLOT = (
    PLOTS_FOLDER
    / "step8_residual_plot.png"
)


# ------------------------------------------------------------
# 2. CREATE FOLDERS IF NEEDED
# ------------------------------------------------------------

OUTPUTS_FOLDER.mkdir(
    exist_ok=True
)

PLOTS_FOLDER.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# 3. LOAD STEP 7 PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print("\n" + "=" * 70)
print("STEP 8 - CALCULATE RESIDUALS")
print("=" * 70)


# ------------------------------------------------------------
# 4. CALCULATE RESIDUALS
# ------------------------------------------------------------

# Formula:
#
# Residual =
# Actual Price - Predicted Price


df["residual"] = (
    df["actual_price_usd"]
    - df["predicted_price_usd"]
)


# ------------------------------------------------------------
# 5. DISPLAY FIRST 10 RESIDUALS
# ------------------------------------------------------------

print("\nFIRST 10 RESIDUALS")
print("=" * 70)


print(
    df[
        [
            "house_id",
            "area_sqft",
            "bedrooms",
            "actual_price_usd",
            "predicted_price_usd",
            "residual",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 6. EXPLAIN ONE RESIDUAL
# ------------------------------------------------------------

example = df.iloc[0]


print("\n" + "=" * 70)
print("ONE RESIDUAL EXPLAINED")
print("=" * 70)


print(
    f"House ID = {example['house_id']}"
)

print(
    f"Area = {example['area_sqft']:,.0f} sq ft"
)

print(
    f"Bedrooms = {example['bedrooms']:,.0f}"
)

print(
    f"Actual Price = "
    f"${example['actual_price_usd']:,.2f}"
)

print(
    f"Predicted Price = "
    f"${example['predicted_price_usd']:,.2f}"
)


print("\nFormula:")

print(
    "Residual = Actual Price - Predicted Price"
)


print("\nCalculation:")

print(
    f"Residual = "
    f"{example['actual_price_usd']:,.2f} "
    f"- {example['predicted_price_usd']:,.2f}"
)


print(
    f"Residual = "
    f"${example['residual']:,.2f}"
)


# ------------------------------------------------------------
# 7. INTERPRET THE RESIDUAL
# ------------------------------------------------------------

print("\nInterpretation:")


if example["residual"] > 0:

    print(
        "Positive residual:"
    )

    print(
        "The actual house price was higher "
        "than the model predicted."
    )


elif example["residual"] < 0:

    print(
        "Negative residual:"
    )

    print(
        "The actual house price was lower "
        "than the model predicted."
    )


else:

    print(
        "Residual = 0"
    )

    print(
        "The model predicted the price exactly."
    )


# ------------------------------------------------------------
# 8. RESIDUAL SUMMARY
# ------------------------------------------------------------

mean_residual = (
    df["residual"].mean()
)

minimum_residual = (
    df["residual"].min()
)

maximum_residual = (
    df["residual"].max()
)


print("\n" + "=" * 70)
print("RESIDUAL SUMMARY")
print("=" * 70)


print(
    f"Mean Residual = "
    f"${mean_residual:,.2f}"
)

print(
    f"Minimum Residual = "
    f"${minimum_residual:,.2f}"
)

print(
    f"Maximum Residual = "
    f"${maximum_residual:,.2f}"
)


# ------------------------------------------------------------
# 9. SAVE RESIDUAL DATA
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 10. CREATE RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    df["predicted_price_usd"],
    df["residual"],
    alpha=0.7,
)


plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2,
)


# ------------------------------------------------------------
# 11. ADD LABELS
# ------------------------------------------------------------

plt.xlabel(
    "Predicted House Price (USD)"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Residual Plot - Multiple Linear Regression"
)

plt.grid(
    alpha=0.3
)


# ------------------------------------------------------------
# 12. SAVE PLOT
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight",
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)


print(
    OUTPUT_FILE
)

print(
    OUTPUT_PLOT
)


# ------------------------------------------------------------
# 13. SHOW PLOT
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 14. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)