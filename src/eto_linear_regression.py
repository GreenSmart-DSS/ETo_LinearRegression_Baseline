"""
Evapotranspiration (ETo) Linear Regression Baseline Modeling Framework.

This module provides a research-grade pipeline for evaluating all possible linear regression
combinations to estimate reference evapotranspiration (ETo). It features:
- Exhaustive feature selection with computational complexity warnings.
- Repeated K-Fold Cross-Validation.
- Distinct statistical (Pearson R^2) and hydrological (NSE) performance metrics.
- Comprehensive multicollinearity analysis using Variance Inflation Factor (VIF).
- Advanced robust input validation and error handling.
- Automated generation of structured Excel reports and separated scientific visualizations.

Author: Morteza Khoshsimaie Chenar: Ph.D. in Irrigation and Drainage Engineering, Department of Irrigation and Reclamation Engineering, University of Tehran
Year: 2026
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Avoid redundant warnings in output
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# LOAD GLOBAL CONFIGURATION FROM JSON
# =====================================================================
# تعیین مسیر فایل تنظیمات (در پوشه اصلی پروژه)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"[CONFIG ERROR] Could not find 'config.json' at {CONFIG_PATH}. "
        f"Please ensure the file exists in the root directory."
    )

DATA_PATH: str = config.get("DATA_PATH", "data/sample_ETo_dataset.xlsx")
TARGET_COLUMN: str = config.get("TARGET_COLUMN", "ETo")
FEATURE_COLUMNS: List[str] = config.get("FEATURE_COLUMNS", [])
CV_SPLITS: int = config.get("CV_SPLITS", 5)
CV_REPEATS: int = config.get("CV_REPEATS", 3)
VIF_THRESHOLD: float = config.get("VIF_THRESHOLD", 5.0)
RANDOM_STATE: int = config.get("RANDOM_STATE", 42)
OUTPUT_PATH: str = config.get("OUTPUT_PATH", "results/generated_results/")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)


# =====================================================================
# METRIC DEFINITIONS (Custom Hydrological and Statistical Functions)
# =====================================================================


def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the statistical Coefficient of Determination (R^2) as the squared
    Pearson correlation coefficient. This measures linear correlation, ignoring bias.
    """
    if len(y_true) < 2:
        return 0.0

    correlation_matrix = np.corrcoef(y_true, y_pred)
    # Check for NaN correlations (constant arrays)
    if np.isnan(correlation_matrix[0, 1]):
        return 0.0
    r = correlation_matrix[0, 1]
    return float(r**2)


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) coefficient.
    In hydrological modeling, NSE evaluates predictive power relative to the mean of observations.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0.0:
        return -np.inf
    return float(1.0 - (numerator / denominator))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a comprehensive set of evaluation metrics."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = coefficient_of_determination(y_true, y_pred)
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "Bias": bias}


# =====================================================================
# MULTICOLLINEARITY ANALYSIS
# =====================================================================


def calculate_vif(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """Calculate the Variance Inflation Factor (VIF) to detect multicollinearity."""
    if len(features) <= 1:
        return {f: 1.0 for f in features}

    sub_df = df[features].dropna()
    sub_df_const = sub_df.copy()
    sub_df_const["intercept"] = 1.0

    vif_dict = {}
    for idx, col in enumerate(features):
        try:
            vif_val = variance_inflation_factor(sub_df_const.values, idx)
            vif_dict[col] = float(vif_val)
        except Exception:
            vif_dict[col] = np.nan

    return vif_dict


# =====================================================================
# CORE PIPELINE CLASS WITH ADVANCED VALIDATION
# =====================================================================


class EToBaselinePipeline:
    """
    Encapsulates the complete data loading, validation, exhaustive modeling,
    and segregated reporting pipeline for ETo baseline regression.
    """

    def __init__(
        self,
        data_path: str,
        target: str,
        features: List[str],
        cv_splits: int,
        cv_repeats: int,
        vif_threshold: float,
        random_state: int,
        output_path: str,
    ):
        self.data_path = data_path
        self.target = target
        self.features = features
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.vif_threshold = vif_threshold
        self.random_state = random_state
        self.output_path = output_path

        # Resolve distinct plots folder path
        self.plots_path = os.path.join(self.output_path, "plots")

        # Initialize placeholders
        self.df: pd.DataFrame = pd.DataFrame()
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series()

    def validate_configuration(self) -> None:
        """Validates critical pipeline settings before execution to prevent downstream crashes."""
        print("[INFO] Validating global configuration parameters...")
        if not isinstance(self.cv_splits, int) or self.cv_splits < 2:
            raise ValueError(
                f"[CONFIG ERROR] CV_SPLITS must be an integer >= 2. Received: {self.cv_splits}"
            )

        if not isinstance(self.cv_repeats, int) or self.cv_repeats < 1:
            raise ValueError(
                f"[CONFIG ERROR] CV_REPEATS must be an integer >= 1. Received: {self.cv_repeats}"
            )

        if (
            not isinstance(self.vif_threshold, (int, float))
            or self.vif_threshold <= 1.0
        ):
            raise ValueError(
                f"[CONFIG ERROR] VIF_THRESHOLD must be a float/int > 1.0. Received: {self.vif_threshold}"
            )

        if not self.features or not isinstance(self.features, list):
            raise ValueError(
                "[CONFIG ERROR] FEATURE_COLUMNS must be a non-empty list of strings."
            )

        if self.target in self.features:
            raise ValueError(
                f"[CONFIG ERROR] TARGET_COLUMN '{self.target}' cannot be included inside FEATURE_COLUMNS."
            )

        # Ensure distinct folders exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def load_and_validate_data(self) -> None:
        """Loads dataset, enforces numerical constraints, and handles missing values."""
        self.validate_configuration()
        print(f"[INFO] Loading data from: {self.data_path}")

        # 1. File existence check
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[DATA ERROR] Cannot find dataset at '{self.data_path}'. Please check file path."
            )

        if self.data_path.endswith(".xlsx"):
            self.df = pd.read_excel(self.data_path)
        elif self.data_path.endswith(".csv"):
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError(
                "[DATA ERROR] Unsupported file format. The pipeline requires either .xlsx or .csv files."
            )

        # 2. Column verification
        required_cols = [self.target] + self.features
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(
                f"[DATA ERROR] The following configured columns are missing from the dataset: {missing_cols} "
                f"Columns actually present inside the file: {list(self.df.columns)}"
            )

        # 3. Numeric type enforcement
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                print(
                    f"[WARNING] Column '{col}' contains non-numeric values. Forcing numeric conversion..."
                )
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # 4. Handle Missing Values
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=required_cols)
        final_len = len(self.df)
        if initial_len != final_len:
            print(
                f"[INFO] Automatically cleaned {initial_len - final_len} rows containing NaN values."
            )

        # 5. Minimum Data Requirement Check for selected CV Splits
        min_required_rows = self.cv_splits * 2
        if final_len < min_required_rows:
            raise ValueError(
                f"[DATA ERROR] Cleansed dataset contains only {final_len} rows. "
                f"Applying {self.cv_splits}-Fold Cross-Validation requires at least {min_required_rows} rows."
            )

        # 6. Zero Variance Check
        for col in self.features:
            if self.df[col].std() == 0:
                raise ValueError(
                    f"[DATA ERROR] Predictor '{col}' has zero variance (constant values). "
                    f"This will crash linear regression models and VIF calculations. Please remove this column."
                )

        self.X = self.df[self.features]
        self.y = self.df[self.target]
        print(
            f"[INFO] Dataset loaded and verified. Final analytical matrix shape: {self.df.shape}"
        )

    def generate_feature_combinations(self) -> List[Tuple[str, ...]]:
        """Generates all non-empty subsets of feature combinations."""
        n = len(self.features)
        total_combinations = (2**n) - 1
        print(f"[INFO] Target variable: {self.target} | Predictor count: {n}")
        print(f"[INFO] Total combination paths to calculate: {total_combinations}")

        if n > 10:
            warnings.warn(
                f"[WARNING] Large feature set (n={n}, combinations={total_combinations}). "
                "The exhaustive search could become computationally heavy. Consider reduction.",
                UserWarning,
            )

        combinations = []
        for r in range(1, n + 1):
            for comb in itertools.combinations(self.features, r):
                combinations.append(comb)
        return combinations

    def run_exhaustive_regression(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs the exhaustive search validation pipeline over all feature combinations."""
        combinations = self.generate_feature_combinations()

        performance_records = []
        coefficients_records = []
        vif_records = []

        rkf = RepeatedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state,
        )

        for idx, combo in enumerate(combinations):
            combo_list = list(combo)
            combo_str = ", ".join(combo_list)
            num_feats = len(combo_list)

            # 1. Repeated K-Fold Cross-Validation
            fold_metrics = []
            for train_idx, test_idx in rkf.split(self.X):
                X_train, X_test = (
                    self.X.iloc[train_idx][combo_list],
                    self.X.iloc[test_idx][combo_list],
                )
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                metrics = compute_metrics(y_test.values, preds)
                fold_metrics.append(metrics)

            # Aggregate CV metrics
            metric_keys = ["RMSE", "MAE", "R2", "NSE", "Bias"]
            aggregated = {}
            for key in metric_keys:
                vals = [m[key] for m in fold_metrics]
                aggregated[f"{key}_mean"] = np.mean(vals)
                aggregated[f"{key}_std"] = np.std(vals)

            # 2. Extract final coefficients trained on full dataset
            full_model = LinearRegression()
            full_model.fit(self.X[combo_list], self.y)

            intercept = full_model.intercept_
            coeffs = full_model.coef_

            terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coeffs, combo_list)]
            eq_str = f"{self.target} = {intercept:.4f} " + " ".join(terms)

            coefficients_records.append(
                {
                    "Features": combo_str,
                    "Intercept": intercept,
                    "Coefficients": str(dict(zip(combo_list, np.round(coeffs, 4)))),
                    "Equation": eq_str,
                }
            )

            # 3. VIF Collinearity Check
            vifs = calculate_vif(self.df, combo_list)
            max_vif = max(vifs.values()) if vifs else 1.0
            usability = "Usable" if max_vif <= self.vif_threshold else "Not usable"

            vif_records.append(
                {
                    "Features": combo_str,
                    "VIF_Values": str({k: round(v, 3) for k, v in vifs.items()}),
                    "Max_VIF": round(max_vif, 3),
                    "Usability": usability,
                }
            )

            # Append Overall metrics
            performance_records.append(
                {"Features": combo_str, "Num_Features": num_feats, **aggregated}
            )

            if (idx + 1) % 10 == 0 or (idx + 1) == len(combinations):
                print(
                    f"[PROGRESS] Evaluated {idx + 1}/{len(combinations)} combinations."
                )

        df_perf = pd.DataFrame(performance_records)
        df_coef = pd.DataFrame(coefficients_records)
        df_vif = pd.DataFrame(vif_records)

        # Merge all data into one master workbook
        df_complete = df_perf.merge(df_coef, on="Features").merge(df_vif, on="Features")

        return df_perf, df_coef, df_vif, df_complete

    def export_results(
        self,
        df_perf: pd.DataFrame,
        df_coef: pd.DataFrame,
        df_vif: pd.DataFrame,
        df_complete: pd.DataFrame,
    ) -> None:
        """Exports dataframes into structured, publication-ready Excel files."""
        print(f"[INFO] Exporting generated scientific tables to: {self.output_path}")
        df_perf.to_excel(
            os.path.join(self.output_path, "performance_results.xlsx"), index=False
        )
        df_coef.to_excel(
            os.path.join(self.output_path, "regression_coefficients.xlsx"), index=False
        )
        df_vif.to_excel(
            os.path.join(self.output_path, "VIF_analysis.xlsx"), index=False
        )
        df_complete.to_excel(
            os.path.join(self.output_path, "complete_linear_regression_results.xlsx"),
            index=False,
        )
        print("[INFO] Excel tables exported successfully.")

    def generate_plots(self, df_complete: pd.DataFrame) -> None:
        """
        Generates and saves research-ready visualizations inside a dedicated plots directory:
        1. Top 10 models ranked by mean RMSE (Only those classified as Usable under VIF constraint).
        2. Comparison of R2 vs NSE across models.
        """
        print(f"[INFO] Generating and saving scientific figures to: {self.plots_path}")

        # Plot 1: Top 10 models based on lowest RMSE (VIF Filtered)
        plt.figure(figsize=(10, 6))
        usable_models = df_complete[df_complete["Usability"] == "Usable"]

        if usable_models.empty:
            print(
                f"[WARNING] No models passed the VIF threshold of {self.vif_threshold}! Plotting top 10 from ALL models instead."
            )
            top_10 = df_complete.sort_values(by="RMSE_mean", ascending=True).head(10)
        else:
            top_10 = usable_models.sort_values(by="RMSE_mean", ascending=True).head(10)

        y_labels = top_10["Features"][::-1]
        x_values = top_10["RMSE_mean"][::-1]
        x_errors = top_10["RMSE_std"][::-1]

        plt.barh(
            y_labels,
            x_values,
            xerr=x_errors,
            color="#2c7fb8",
            capsize=5,
            alpha=0.85,
            edgecolor="black",
        )

        title = "Top 10 Linear Models (CV Mean RMSE)"
        if not usable_models.empty:
            title += f"\nFiltered for VIF <= {self.vif_threshold}"

        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("Mean RMSE (mm/day)", fontsize=10)
        plt.ylabel("Meteorological Predictors", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "top_10_rmse_models.png"), dpi=300)
        plt.close()

        # Plot 2: Relationship between R2 and NSE under CV
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df_complete["R2_mean"],
            df_complete["NSE_mean"],
            alpha=0.7,
            color="teal",
            edgecolor="k",
        )

        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, "r--", alpha=0.75, label="1:1 Equality")
        plt.title(
            "Statistical $R^2$ vs. Hydrological Nash-Sutcliffe Efficiency (NSE)",
            fontsize=11,
            fontweight="bold",
        )
        plt.xlabel("Mean Coefficient of Determination ($R^2$)", fontsize=10)
        plt.ylabel("Mean Nash-Sutcliffe Efficiency (NSE)", fontsize=10)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "R2_vs_NSE_comparison.png"), dpi=300)
        plt.close()

    def generate_residuals_plot(self, df_complete: pd.DataFrame) -> None:
        """Generates a residual plot for the best performing usable model."""
        usable_models = df_complete[df_complete["Usability"] == "Usable"]
        if usable_models.empty:
            print(
                "[WARNING] Skipping residual plot. No models passed the VIF threshold criteria."
            )
            return

        best_model_row = usable_models.sort_values(by="RMSE_mean").iloc[0]
        best_features = best_model_row["Features"].split(", ")

        best_model = LinearRegression()
        best_model.fit(self.X[best_features], self.y)
        predictions = best_model.predict(self.X[best_features])
        residuals = self.y - predictions

        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.6, edgecolors="k", color="coral")
        plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
        plt.title(
            f"Residual Analysis for Best Usable Model\n({', '.join(best_features)})",
            fontsize=11,
            fontweight="bold",
        )
        plt.xlabel("Predicted ETo (mm/day)", fontsize=10)
        plt.ylabel("Residuals (Observed - Predicted)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, "best_model_residuals.png"), dpi=300)
        plt.close()
        print("[INFO] Residual analysis plot exported successfully.")


# =====================================================================
# PIPELINE EXECUTION
# =====================================================================

if __name__ == "__main__":

    try:
        # Initialize pipeline with configurable global variables
        pipeline = EToBaselinePipeline(
            data_path=DATA_PATH,
            target=TARGET_COLUMN,
            features=FEATURE_COLUMNS,
            cv_splits=CV_SPLITS,
            cv_repeats=CV_REPEATS,
            vif_threshold=VIF_THRESHOLD,
            random_state=RANDOM_STATE,
            output_path=OUTPUT_PATH,
        )

        # Load, validate and execute
        pipeline.load_and_validate_data()
        df_perf, df_coef, df_vif, df_complete = pipeline.run_exhaustive_regression()

        # Save structured files
        pipeline.export_results(df_perf, df_coef, df_vif, df_complete)
        pipeline.generate_plots(df_complete)
        pipeline.generate_residuals_plot(df_complete)

        # Summary Console Output
        usable_df = df_complete[df_complete["Usability"] == "Usable"]

        print("\n" + "=" * 50)
        print(" PIPELINE RUN COMPLETED SUCCESSFULLY ")
        print("=" * 50)

        if not usable_df.empty:
            best_model = usable_df.sort_values(by="RMSE_mean").iloc[0]
            print(
                f"Optimal Usable Model (VIF <= {VIF_THRESHOLD}):\n"
                f"Features: {best_model['Features']}\n"
                f"Equation: {best_model['Equation']}\n"
                f"CV RMSE:  {best_model['RMSE_mean']:.4f} mm/day\n"
                f"CV NSE:   {best_model['NSE_mean']:.4f}"
            )
        else:
            print(f"[WARNING] No model met the VIF threshold of {VIF_THRESHOLD}.")

        print(f"\nAll Excel tables saved at: {OUTPUT_PATH}")
        print(f"All graphic plots saved at: {pipeline.plots_path}")
        print("=" * 50)

    except Exception as e:
        print("\nFATAL PIPELINE ERROR] Execution halted:")
        print(f"-> {type(e).__name__}: {str(e)}")
