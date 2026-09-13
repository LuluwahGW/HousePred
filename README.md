# House Price Prediction

A machine learning model that predicts house prices using the Kaggle "House Prices: Advanced Regression Techniques" dataset.

## Overview

This project builds a Linear Regression model to estimate house sale prices based on 79 features describing residential homes in Ames, Iowa.

**Dataset:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## What it does

1. **Data Analysis** – Checks for missing values and visualizes the relationship between overall quality and sale price.
2. **Data Preprocessing** – Fills missing numerical values with the mean, fills missing categorical values with the mode, and drops columns with too many missing values (`Alley`, `PoolQC`, `Fence`, `MiscFeature`, `FireplaceQu`).
3. **Feature Selection** – One-hot encodes categorical features, removes duplicate columns, and scales all features with `StandardScaler`.
4. **Model Training** – Trains a `LinearRegression` model.
5. **Model Evaluation** – Evaluates performance using 5-fold cross-validation (R², MAE, RMSE).

## Results

| Metric | Score |
|--------|-------|
| R²     | 0.68  |
| RMSE   | ~44,900 |
| MAE    | ~19,150 |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

1. Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) into the project folder.
2. Run the script:

```bash
python house_price_prediction.py
```

Plots are saved as PNG files in the project folder.

## Notes / Next Steps

- Linear Regression is used as a baseline; tree-based models (e.g. Random Forest, XGBoost) would likely perform better on this tabular data.
- Further improvements could include feature engineering, outlier removal, and log-transforming the target variable.
