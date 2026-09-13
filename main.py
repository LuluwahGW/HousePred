# =====================================================================
# House Prices Prediction Using Regression
# Kaggle: House Prices - Advanced Regression Techniques
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---
# (SETUP: loading the data from csv files)
# ---
print("Loading data")
try:
    train_data_raw = pd.read_csv('train.csv', index_col='Id')
    test_data = pd.read_csv('test.csv', index_col='Id')
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    exit()

print("\n--- 5. Dataset Sample ---")
print("Displaying head() of the *original* raw training data:")
print(train_data_raw.head())


train_data = train_data_raw.copy()
y_train_full = train_data.pop('SalePrice')  
all_data = pd.concat([train_data, test_data])
print(f"Combined data shape: {all_data.shape}")


# ---
# 2. DATA ANALYSIS
# ---
print("\n--- 2. Data Analysis ---")
# 2.1: Check for nulls
print("\nChecking for Null Values:")
nulls = all_data.isnull().sum()
nulls = nulls[nulls > 0].sort_values(ascending=False)
print(nulls.head(5))


print("\nPlotting Quality vs. Price...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_data_raw['OverallQual'], y=y_train_full, alpha=0.6, color='steelblue')
plt.title('Data Analysis: Overall Quality vs. SalePrice (Scatter Plot)')
plt.ylabel('SalePrice (Dollars)')
plt.xlabel('Overall Quality')
plt.savefig('qual_vs_price_scatter.png')
print("Saved 'qual_vs_price_scatter.png'")


# ---
# 3. DATA PREPROCESSING
# ---
print("\n--- 3. Data Preprocessing ---")
# 3.1: Visualize missing data + info()
print("\nDataFrame Info (on combined data):")
all_data.info()
plt.figure(figsize=(15, 7))
sns.heatmap(all_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Data Preprocessing: Heatmap of Missing Data (BEFORE Filling)')
plt.savefig('missing_data_heatmap_before.png')
print("\nSaved 'missing_data_heatmap_before.png'")


print("\nFilling numerical data with MEAN...")
col_to_plot = 'LotFrontage'
numerical_cols = all_data.select_dtypes(include=np.number).columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(all_data[col_to_plot], kde=True, ax=ax1, color='blue')
ax1.set_title(f'Before: {col_to_plot}')
for col in numerical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mean())
sns.histplot(all_data[col_to_plot], kde=True, ax=ax2, color='green')
ax2.set_title(f'After: {col_to_plot} (Mean Filled)')
plt.savefig('mean_fill_before_after.png')
print("Saved 'mean_fill_before_after.png'")

# 3.3: Data filling (Mode) + Dropping columns
print("\nFilling categorical data with MODE and dropping high-null cols...")
cols_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']
all_data = all_data.drop(columns=cols_to_drop)
print(f"Dropped columns: {cols_to_drop}")
categorical_cols = all_data.select_dtypes(include='object').columns
for col in categorical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
print("Filled categorical columns with mode.")

# 3.4: Heatmap after filling missing values
print("\nCreating heatmap AFTER filling missing values...")
plt.figure(figsize=(15, 7))
sns.heatmap(all_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Data Preprocessing: Heatmap of Missing Data (AFTER Filling - Should Be Empty)')
plt.savefig('missing_data_heatmap_after.png')
print("Saved 'missing_data_heatmap_after.png'")

# Check if any nulls remain
remaining_nulls = all_data.isnull().sum().sum()
print(f"\nRemaining null values after preprocessing: {remaining_nulls}")


# ---
# 4. FEATURE SELECTION
# ---
print("\n--- 4. Feature Selection ---")
# 4.1: One-Hot Encoding (converting categorical to binary)
print("Converting categorical values to binary (One-Hot Encoding)...")
all_data_ohc = pd.get_dummies(all_data, drop_first=True, dummy_na=False)
print(f"Data shape before OHC: {all_data.shape}")
print(f"Data shape after OHC: {all_data_ohc.shape}")

# 4.2: Remove duplicated columns
print("\nRemoving duplicated columns...")
all_data_ohc = all_data_ohc.loc[:, ~all_data_ohc.columns.duplicated()]
print(f"Data shape after OHC (deduped): {all_data_ohc.shape}")

# 4.3: Scaling (to reduce biased and inaccurate predictions)
print("\nScaling all features (StandardScaler)...")
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data_ohc)
all_data_scaled_df = pd.DataFrame(all_data_scaled, columns=all_data_ohc.columns,
                                   index=all_data_ohc.index)


# ---
# 5. DATA SPLITTING
# ---
print("\n--- 5. Data Splitting ---")
# Split processed (scaled) data back into Train and Test
X_train_processed = all_data_scaled_df.loc[train_data.index]
X_test_processed = all_data_scaled_df.loc[test_data.index]

# Create the validation split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train_full, test_size=0.2, random_state=0)

print(f"Training split: {len(X_train_split)}, Validation split: {len(X_val)}")


# ---
# 6. MODEL TRAINING
# ---
print("\n--- 6. Model Training (Linear Regression) ---")
# Linear Regression Model
model = LinearRegression()
# We will train on full data for cross-validation
model.fit(X_train_processed, y_train_full)
print("Model trained on full training data for cross-validation.")


# ---
# 7. MODEL EVALUATION (K-Fold Cross-Validation)
# ---
print("\n--- 7. Model Evaluation (5-Fold Cross-Validation) ---")
print("Running 5-fold cross-validation on the entire training set...")

cv_model = LinearRegression()

# Perform 5-fold validation and get predictions for plotting
cv_predictions = cross_val_predict(cv_model, X_train_processed, y_train_full, cv=5)

# Perform 5-fold validation for metrics
r2_scores = cross_val_score(cv_model, X_train_processed, y_train_full,
                             cv=5, scoring='r2')

mse_scores = cross_val_score(cv_model, X_train_processed, y_train_full,
                              cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)

mae_scores = cross_val_score(cv_model, X_train_processed, y_train_full,
                              cv=5, scoring='neg_mean_absolute_error')
mae_scores = -mae_scores

print("Cross-Validation complete. Averaging the 5 folds:")
print(f"\n  Average R-Squared (R²): {r2_scores.mean():.4f} (Std: {r2_scores.std():.4f})")
print(f"  Average MAE: {mae_scores.mean():.4f} (Std: {mae_scores.std():.4f})")
print(f"  Average RMSE: {rmse_scores.mean():.4f} (Std: {rmse_scores.std():.4f})")

# Calculate overall metrics from cross-validation predictions
cv_r2 = r2_score(y_train_full, cv_predictions)
cv_mae = mean_absolute_error(y_train_full, cv_predictions)
cv_rmse = np.sqrt(mean_squared_error(y_train_full, cv_predictions))

print("\nOverall metrics from cross-validation predictions:")
print(f"  R-Squared (R²): {cv_r2:.4f}")
print(f"  MAE: {cv_mae:.4f}")
print(f"  RMSE: {cv_rmse:.4f}")


# ---
# (FINAL SUBMISSION)
# ---
print("\n--- Retraining on full data for submission ---")
model.fit(X_train_processed, y_train_full)
final_preds_dollars = model.predict(X_test_processed)

final_preds_dollars = np.maximum(final_preds_dollars, 0)
print(f"Minimum prediction value: {final_preds_dollars.min():.2f} (negative values set to 0)")

print("\n--- ALL DONE ---")
print("Your plots are saved as PNG files.")

plt.show()
