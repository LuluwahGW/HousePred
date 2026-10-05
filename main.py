# House Prices - visual walkthrough
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict

os.makedirs("plots", exist_ok=True)       # create output folder
sns.set_theme(style="whitegrid")          # global plot style


def save(name):
    plt.tight_layout()                    # fix spacing
    plt.savefig(f"plots/{name}.png", dpi=120)   # save as image
    plt.close()                           # free memory
    print(f"saved plots/{name}.png")


# 0. LOAD DATA
train = pd.read_csv("train.csv", index_col="Id")   # read CSV table
test = pd.read_csv("test.csv", index_col="Id")
print("train shape:", train.shape, "| test shape:", test.shape)


# PLOT 1: target before/after log
fig, axes = plt.subplots(1, 2, figsize=(12, 4))    # two side-by-side plots
sns.histplot(train["SalePrice"], kde=True, ax=axes[0], color="steelblue")   # distribution histogram
axes[0].set_title("SalePrice (skewed)")
sns.histplot(np.log1p(train["SalePrice"]), kde=True, ax=axes[1], color="seagreen")   # log reduces skew
axes[1].set_title("log(SalePrice) (bell curve)")
save("01_target_before_after_log")


# PLOT 2: missing values per column
missing = train.drop(columns="SalePrice").isnull().sum()    # count missing cells
missing = missing[missing > 0].sort_values(ascending=False).head(20)   # top 20 columns
plt.figure(figsize=(9, 6))
sns.barplot(x=missing.values, y=missing.index, color="indianred")   # horizontal bar chart
plt.title("Top 20 columns with missing values")
plt.xlabel("Number of missing rows")
save("02_missing_values")


# PLOT 3: outliers (big house, low price)
is_outlier = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)   # outlier condition
plt.figure(figsize=(8, 5))
plt.scatter(train.loc[~is_outlier, "GrLivArea"], train.loc[~is_outlier, "SalePrice"],
            alpha=0.5, label="normal", color="steelblue")   # normal houses
plt.scatter(train.loc[is_outlier, "GrLivArea"], train.loc[is_outlier, "SalePrice"],
            color="red", s=80, label="outliers (removed)")  # outlier houses
plt.xlabel("GrLivArea (sq ft)")
plt.ylabel("SalePrice")
plt.title("Outliers: big houses, low prices")
plt.legend()                                        # show label box
save("03_outliers")
train = train[~is_outlier]                          # drop outlier rows


# PLOT 4: correlation with price
numeric = train.select_dtypes(include=np.number)    # numeric columns only
corr_with_price = numeric.corr()["SalePrice"].drop("SalePrice").sort_values()   # correlation with price
top = pd.concat([corr_with_price.head(5), corr_with_price.tail(10)])   # lowest + highest
plt.figure(figsize=(8, 6))
sns.barplot(x=top.values, y=top.index, palette="coolwarm", hue=top.index, legend=False)   # colored bar chart
plt.title("Correlation with SalePrice")
save("04_correlation_with_price")


# PLOT 5: correlation heatmap
top_cols = numeric.corr()["SalePrice"].abs().sort_values(ascending=False).head(10).index   # top 10 features
plt.figure(figsize=(9, 7))
sns.heatmap(numeric[top_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)   # colored correlation grid
plt.title("Correlation heatmap (top 10 features)")
save("05_correlation_heatmap")


# PLOT 6-7: price by category
order = train.groupby("Neighborhood")["SalePrice"].median().sort_values().index   # sort by median price
plt.figure(figsize=(12, 5))
sns.boxplot(data=train, x="Neighborhood", y="SalePrice", order=order, color="lightsteelblue")   # price spread per group
plt.xticks(rotation=60)                             # rotate labels
plt.title("SalePrice by neighborhood")
save("06_price_by_neighborhood")

plt.figure(figsize=(8, 5))
sns.boxplot(data=train, x="OverallQual", y="SalePrice", color="lightsteelblue")
plt.title("SalePrice by overall quality")
save("07_price_by_quality")


# PREPARE DATA FOR MODELS
def add_features(df):
    df = df.copy()                                  # avoid changing original
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]   # total square feet
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]   # age when sold
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0))   # combined bathrooms
    return df


X = add_features(train.drop(columns="SalePrice"))   # features table
y = np.log1p(train["SalePrice"])                    # log of price
X_test = add_features(test)

num_cols = X.select_dtypes(include=np.number).columns   # numeric column names
cat_cols = X.select_dtypes(include="object").columns    # text column names

preprocess = ColumnTransformer([                    # different cleaning per type
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")),     # fill gaps: median
                      ("scale", StandardScaler())]), num_cols),        # same scale
    ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="None")),   # fill gaps: "None"
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),    # text to 0/1 columns
])


# PLOT 8: compare models (5-fold cross-validation)
models = {
    "Ridge": Ridge(alpha=10),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=0),
}
results = {}
for name, model in models.items():
    pipe = Pipeline([("pre", preprocess), ("model", model)])   # clean then model
    scores = -cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")   # 5-fold error scores
    results[name] = scores
    print(f"{name}: RMSE = {scores.mean():.4f} (+/- {scores.std():.4f})")

plt.figure(figsize=(7, 4))
means = [s.mean() for s in results.values()]
stds = [s.std() for s in results.values()]
plt.bar(list(results.keys()), means, yerr=stds, capsize=6, color="steelblue")   # bars with error whiskers
plt.ylabel("CV RMSE (log price), lower is better")
plt.title("Model comparison")
save("08_model_comparison")


# PLOT 9: predicted vs actual
best_name = min(results, key=lambda k: results[k].mean())   # lowest error model
print("Best model:", best_name)
best_pipe = Pipeline([("pre", preprocess), ("model", models[best_name])])
pred_log = cross_val_predict(best_pipe, X, y, cv=5)   # out-of-fold predictions
pred = np.expm1(pred_log)                           # back to dollars
actual = np.expm1(y)

plt.figure(figsize=(6, 6))
plt.scatter(actual, pred, alpha=0.4, color="steelblue")   # one dot per house
lims = [actual.min(), actual.max()]
plt.plot(lims, lims, color="red", linestyle="--", label="perfect prediction")   # diagonal reference line
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title(f"Predicted vs actual ({best_name})")
plt.legend()
save("09_predicted_vs_actual")


# PLOT 10: residuals (errors)
residuals = y - pred_log                            # actual minus predicted
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(pred_log, residuals, alpha=0.4, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")     # zero-error line
axes[0].set_xlabel("Predicted log price")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs predictions")
sns.histplot(residuals, kde=True, ax=axes[1], color="seagreen")   # error distribution
axes[1].set_title("Residual distribution")
save("10_residuals")


# PLOT 11: feature importance
final_pipe = Pipeline([("pre", preprocess), ("model", models["GradientBoosting"])]).fit(X, y)   # train on all data
names = final_pipe.named_steps["pre"].get_feature_names_out()   # column names after encoding
importances = pd.Series(final_pipe.named_steps["model"].feature_importances_, index=names)   # importance per feature
importances = importances.sort_values().tail(15)    # top 15
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Top 15 most important features")
save("11_feature_importance")


# SUBMISSION FILE
test_pred = np.expm1(final_pipe.predict(X_test))    # predict, back to dollars
pd.DataFrame({"Id": X_test.index, "SalePrice": test_pred}).to_csv("submission.csv", index=False)   # save Kaggle file
print("saved submission.csv")
print("Done. Open the 'plots' folder.")