# Miami Housing Price Analysis

A machine learning project that explores Miami housing sale data, analyzes price trends, and builds decision tree / random forest models to predict home sale prices.

## Overview

This project uses `pandas` and `scikit-learn` to:

- Load and clean a Miami housing dataset (`miami-housing.csv`)
- Explore key statistics on housing features and sale prices
- Train a **Decision Tree Regressor** to predict `SALE_PRC` (sale price)
- Tune the tree's `max_leaf_nodes` to find the best trade-off between underfitting and overfitting
- Train a **Random Forest Regressor** to improve accuracy and reduce overfitting compared to the single decision tree
- Compare actual vs. predicted prices and evaluate both models using **Mean Absolute Error (MAE)**

## Dataset

`miami-housing.csv` contains property records for Miami-Dade County, including:

| Feature | Description |
|---|---|
| `LND_SQFOOT` | Land size in square feet |
| `TOT_LVG_AREA` | Total living area in square feet |
| `SPEC_FEAT_VAL` | Value of special features |
| `age` | Age of the property |
| `LATITUDE` | Property latitude |
| `LONGITUDE` | Property longitude |
| `SALE_PRC` | Sale price (target variable) |

Rows with missing values are dropped before modeling.

The script will print:

1. Summary statistics of the dataset
2. Sample predictions from the initial Decision Tree model
3. Actual vs. predicted prices for a handful of validation rows
4. MAE for the Decision Tree model
5. A comparison of MAE across different `max_leaf_nodes` values (5, 50, 500, 5000)
6. Training and validation MAE for the Random Forest model
7. Actual vs. predicted prices for the Random Forest model

## Modeling Approach

1. **Baseline Decision Tree** — `DecisionTreeRegressor(max_depth=10, min_samples_leaf=10)` fit on a 75/25 train/validation split.
2. **Leaf Node Tuning** — MAE is compared across several `max_leaf_nodes` values to find the setting that best balances bias and variance (500 performed best in testing).
3. **Random Forest** — `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5)` is trained to reduce overfitting and improve generalization over the single decision tree.

## Results

The Random Forest model achieves a lower validation MAE than the single Decision Tree, indicating better predictive performance and less overfitting.

## Future Improvements

- Cross-validation for more robust performance estimates
- Additional feature engineering (e.g., distance to coast, neighborhood encoding)
- Hyperparameter tuning via grid search
- Comparison with other models (e.g., Gradient Boosting, XGBoost)
