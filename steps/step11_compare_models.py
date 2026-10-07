import pandas as pd
from pathlib import Path


# ============================================================
# STEP 11 - COMPARE SIMPLE VS MULTIPLE LINEAR REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

CURRENT_PROJECT = Path(__file__).resolve().parents[1]

DESKTOP = CURRENT_PROJECT.parent


SIMPLE_METRICS_FILE = (
    DESKTOP
    / "house-price-linear-regression"
    / "outputs"
    / "step10_metrics.csv"
)


MULTIPLE_METRICS_FILE = (
    CURRENT_PROJECT
    / "outputs"
    / "step10_metrics.csv"
)


OUTPUT_FILE = (
    CURRENT_PROJECT
    / "outputs"
    / "step11_model_comparison.csv"
)


# ------------------------------------------------------------
# 2. LOAD SIMPLE REGRESSION METRICS
# ------------------------------------------------------------

simple_metrics = pd.read_csv(
    SIMPLE_METRICS_FILE
)


# ------------------------------------------------------------
# 3. LOAD MULTIPLE REGRESSION METRICS
# ------------------------------------------------------------

multiple_metrics = pd.read_csv(
    MULTIPLE_METRICS_FILE
)


print("\n" + "=" * 70)
print("STEP 11 - COMPARE SIMPLE VS MULTIPLE REGRESSION")
print("=" * 70)


# ------------------------------------------------------------
# 4. CONVERT METRICS INTO DICTIONARIES
# ------------------------------------------------------------

simple = dict(
    zip(
        simple_metrics["metric"],
        simple_metrics["value"],
    )
)


multiple = dict(
    zip(
        multiple_metrics["metric"],
        multiple_metrics["value"],
    )
)


# ------------------------------------------------------------
# 5. GET IMPORTANT METRICS
# ------------------------------------------------------------

simple_mae = simple["MAE"]
simple_rmse = simple["RMSE"]
simple_r2 = simple["R_squared"]


multiple_mae = multiple["MAE"]
multiple_rmse = multiple["RMSE"]
multiple_r2 = multiple["R_squared"]


# ------------------------------------------------------------
# 6. DISPLAY SIMPLE REGRESSION RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIMPLE LINEAR REGRESSION")
print("=" * 70)

print("Feature:")
print("area_sqft")

print(
    f"\nMAE  = ${simple_mae:,.2f}"
)

print(
    f"RMSE = ${simple_rmse:,.2f}"
)

print(
    f"R²   = {simple_r2:.4f}"
)


# ------------------------------------------------------------
# 7. DISPLAY MULTIPLE REGRESSION RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 70)

print("Features:")
print("area_sqft")
print("bedrooms")

print(
    f"\nMAE  = ${multiple_mae:,.2f}"
)

print(
    f"RMSE = ${multiple_rmse:,.2f}"
)

print(
    f"R²   = {multiple_r2:.4f}"
)


# ------------------------------------------------------------
# 8. CALCULATE DIFFERENCES
# ------------------------------------------------------------

mae_difference = (
    simple_mae
    - multiple_mae
)


rmse_difference = (
    simple_rmse
    - multiple_rmse
)


r2_difference = (
    multiple_r2
    - simple_r2
)


# ------------------------------------------------------------
# 9. DISPLAY DIFFERENCES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIFFERENCE")
print("=" * 70)


print(
    f"MAE improvement = "
    f"${mae_difference:,.2f}"
)

print(
    f"RMSE improvement = "
    f"${rmse_difference:,.2f}"
)

print(
    f"R² change = "
    f"{r2_difference:.4f}"
)


# ------------------------------------------------------------
# 10. DECIDE WHICH MODEL IS BETTER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


better_count = 0


if multiple_mae < simple_mae:

    print(
        "MAE: Multiple regression is better."
    )

    better_count += 1

else:

    print(
        "MAE: Simple regression is better."
    )


if multiple_rmse < simple_rmse:

    print(
        "RMSE: Multiple regression is better."
    )

    better_count += 1

else:

    print(
        "RMSE: Simple regression is better."
    )


if multiple_r2 > simple_r2:

    print(
        "R²: Multiple regression is better."
    )

    better_count += 1

else:

    print(
        "R²: Simple regression is better."
    )


# ------------------------------------------------------------
# 11. FINAL CONCLUSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)


if better_count == 3:

    print(
        "Adding bedrooms improved the model "
        "on all three evaluation metrics."
    )


elif better_count >= 2:

    print(
        "Adding bedrooms improved the model overall."
    )


elif better_count == 1:

    print(
        "Adding bedrooms produced only limited improvement."
    )


else:

    print(
        "Adding bedrooms did not improve the model."
    )


# ------------------------------------------------------------
# 12. CREATE COMPARISON TABLE
# ------------------------------------------------------------

comparison = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "RMSE",
            "R_squared",
        ],
        "simple_regression": [
            simple_mae,
            simple_rmse,
            simple_r2,
        ],
        "multiple_regression": [
            multiple_mae,
            multiple_rmse,
            multiple_r2,
        ],
    }
)


# ------------------------------------------------------------
# 13. DISPLAY COMPARISON TABLE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)


print(
    comparison
)


# ------------------------------------------------------------
# 14. SAVE COMPARISON
# ------------------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("COMPARISON FILE SAVED")
print("=" * 70)


print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 70)