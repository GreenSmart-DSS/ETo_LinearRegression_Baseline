# Reference Evapotranspiration (ETo) Linear Regression Baseline Modeling

An open-source, robust, and research-grade framework for evaluating exhaustive feature combinations of meteorological parameters to estimate Reference Evapotranspiration ($ETo$).

## 1. Project Overview

Reference evapotranspiration ($ETo$) modeling is critical for agricultural water management and irrigation engineering. While advanced machine learning (ML) architectures offer high predictive capacities, they often lack interpretability and demand significant computational power.

This repository provides a **transparent baseline modeling framework** using Multiple Linear Regression. It is designed to strictly evaluate meteorological predictors, analyze multicollinearity, and establish a rigorous performance threshold before deploying complex ML algorithms.

### Key Features

* **Exhaustive Feature Selection:** Automatically generates and evaluates all possible combinations ($2^n - 1$) of input variables.
* **Robust Error Handling:** Advanced input validation ensures data integrity, checks for zero-variance predictors, and safely handles missing values.
* **VIF-Filtered Visualizations:** Automatically filters out models suffering from severe multicollinearity before plotting top-performing equations.
* **Separated Outputs:** Neatly organizes tabular data (Excel) and scientific figures (PNG) into distinct output directories.
* **Hydrological vs. Statistical Metrics:** Distinct evaluation of Pearson's $R^2$ and Nash-Sutcliffe Efficiency ($NSE$) under Repeated K-Fold Cross-Validation.
* **Dynamic Configuration:** Entirely driven by an external JSON configuration file, requiring zero modifications to the core Python source code.

## 2. Mathematical Foundations

### 2.1. Standard Baseline Equation

The framework attempts to model the energy-aerodynamic balance through linear combinations of meteorological predictors:

$$ETo \approx \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k + \epsilon$$

### 2.2. Distinct Evaluation Metrics

Standard statistical packages often conflate $R^2$ and $NSE$, but in cross-validated or out-of-sample data, they represent different concepts:

* **Coefficient of Determination ($R^2$):** Calculated as the squared Pearson correlation coefficient. It measures linear correlation (collinearity) and is strictly bounded between 0 and 1, ignoring systematic bias:

$$R^2 = \left( \frac{\sum (y_{obs} - \bar{y}_{obs})(y_{sim} - \bar{y}_{sim})}{\sqrt{\sum (y_{obs} - \bar{y}_{obs})^2 \sum (y_{sim} - \bar{y}_{sim})^2}} \right)^2$$


* **Nash-Sutcliffe Efficiency ($NSE$):** A strict hydrological metric evaluating predictive power relative to the observed mean. It severely penalizes systematic bias and can yield negative values if the model performs worse than the mean baseline:

$$NSE = 1 - \frac{\sum (y_{obs} - y_{sim})^2}{\sum (y_{obs} - \bar{y}_{obs})^2}$$



### 2.3. Multicollinearity Diagnostics (VIF)

Meteorological variables (e.g., temperatures, humidity, radiation) are often heavily correlated. High multicollinearity inflates coefficient variances, destroying the physical interpretability of the regression equation. We compute the Variance Inflation Factor ($VIF$) for each predictor $j$:

$$VIF_j = \frac{1}{1 - R_j^2}$$

* **$VIF \le 5.0$**: Model is scientifically usable (low collinearity).
* **$VIF > 5.0$**: Model suffers from severe collinearity and is flagged as not usable.

## 3. Repository Structure

```text
ETo_LinearRegression_Baseline/
├── README.md
├── requirements.txt
├── LICENSE
├── config.json
├── data/
│   └── sample_ETo_dataset.xlsx
├── src/
│   └── eto_linear_regression.py
└── results/
    └── generated_results/
        ├── performance_results.xlsx
        ├── regression_coefficients.xlsx
        ├── VIF_analysis.xlsx
        ├── complete_linear_regression_results.xlsx
        └── plots/
            ├── top_10_rmse_models.png
            ├── R2_vs_NSE_comparison.png
            └── best_model_residuals.png

```

## 4. Installation and Setup

**Prerequisites:** Python 3.8+

**Step 1: Clone the Repository**

```bash
git clone https://github.com/GreenSmart-DSS/ETo_LinearRegression_Baseline.git
cd ETo_LinearRegression_Baseline

```

**Step 2: Create Virtual Environment & Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

## 5. Usage & Configuration

> ⚠️ **CRITICAL REQUIREMENT:** Do **NOT** modify the core Python code inside `src/eto_linear_regression.py`. All user-defined settings and parameters must be configured exclusively inside the `config.json` file located in the root directory.
> 🛑 **JSON Format Rule:** Standard JSON files do **NOT** support comments (e.g., `//` or `#`). Adding any text outside of the strict key-value JSON syntax will cause the Python parser to fail.

### 5.1. Configure your inputs in `config.json`

To adapt the pipeline to your own custom dataset, simply open the `config.json` file in any text editor and change the values.

**Example Configuration:**

```json
{
    "DATA_PATH": "data/sample_ETo_dataset.xlsx",
    "TARGET_COLUMN": "ETo",
    "FEATURE_COLUMNS": [
        "Tmax",
        "Tmin",
        "RH",
        "Rs",
        "u2",
        "VPD"
    ],
    "CV_SPLITS": 5,
    "CV_REPEATS": 3,
    "VIF_THRESHOLD": 5.0,
    "RANDOM_STATE": 42,
    "OUTPUT_PATH": "results/generated_results/"
}

```

### 5.2. Parameter Settings Reference Guide

| Parameter Key | Expected Data Type | Default Value | Purpose & Customization Instructions |
| --- | --- | --- | --- |
| `DATA_PATH` | `String` | `"data/sample_ETo_dataset.xlsx"` | **Change this** to the path of your own dataset. It supports both Excel (`.xlsx`) and CSV (`.csv`) formats. |
| `TARGET_COLUMN` | `String` | `"ETo"` | **Change this** to match the exact column name of the target variable (dependent variable) in your file. |
| `FEATURE_COLUMNS` | `List of Strings` | `["Tmax", "Tmin", ...]` | **Change this** list to include the exact column names of the meteorological features you wish to evaluate in exhaustive combinations. |
| `CV_SPLITS` | `Integer` | `5` | The number of folds to use for K-Fold cross-validation. Must be $\ge 2$. Recommended: `5` or `10`. |
| `CV_REPEATS` | `Integer` | `3` | The number of times the cross-validation process will be repeated with different random splits. Must be $\ge 1$. |
| `VIF_THRESHOLD` | `Float` | `5.0` | The Variance Inflation Factor limit. Models containing any predictor exceeding this value will be labeled "Not usable" due to high multicollinearity. |
| `RANDOM_STATE` | `Integer` | `42` | A static seed to control random splits. Keep this constant to ensure reproducibility of your statistical results. |
| `OUTPUT_PATH` | `String` | `"results/generated_results/"` | The directory path where all final output Excel files and the `plots/` subfolder will be created. |

### 5.3. Execution

Once your JSON file is configured, run the pipeline directly from your terminal:

```bash
python src/eto_linear_regression.py

```

## 6. Generated Scientific Outputs

Upon successful execution, the framework automatically generates:

1. **Excel Reports (`results/generated_results/`)**:
* `performance_results.xlsx`: Exhaustive evaluation metrics (RMSE, MAE, Bias, $R^2$, $NSE$).
* `regression_coefficients.xlsx`: Explicit mathematical equations and coefficients for every combination.
* `VIF_analysis.xlsx`: Collinearity diagnostics for model usability.
* `complete_linear_regression_results.xlsx`: A master dataset containing all above data.


2. **Visualizations (`results/generated_results/plots/`)**:
* `top_10_rmse_models.png`: Horizontal bar chart of the best models (strictly filtered for $VIF \le 5.0$).
* `R2_vs_NSE_comparison.png`: Scatter plot tracking the mathematical divergence between statistical correlation and hydrological accuracy.
* `best_model_residuals.png`: Residual distribution plot to verify homoscedasticity and linear assumptions of the best usable model.



## 7. Citation

If this baseline framework assists your research, please cite the repository:

```bibtex
@misc{eto_lr_baseline_2026,
  author = {Morteza Khoshsimaie Chenar},
  title = {Reference Evapotranspiration (ETo) Linear Regression Baseline Modeling Framework},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/GreenSmart-DSS/ETo_LinearRegression_Baseline}}
}

```

---
