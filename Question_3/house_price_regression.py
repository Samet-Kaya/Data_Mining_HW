# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Load the Dataset
# ------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
os.chdir(script_dir)  # Change the working directory to the script's location

# Load the dataset
data = pd.read_csv("house_price.csv")  # Replace with your file path

# Display the first few rows
print("First five rows of the dataset:")
print(data.head())

# Display dataset statistics
print("\nDataset statistics:")
print(data.describe())

# ------------------------------
# Step 2: Data Preprocessing
# ------------------------------

# Check for missing values
print("\nMissing values in the dataset:")
missing_values = data.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])

# Drop columns with excessive missing values (>30%)
data = data.drop(columns=missing_values[missing_values > 0.3 * len(data)].index)

# Fill remaining missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill categorical with mode
    else:
        data[col] = data[col].fillna(data[col].median())  # Fill numerical with median

# Detect and handle outliers in the target variable (SalePrice)
sns.boxplot(x=data['SalePrice'])
plt.title("Boxplot of Target Variable (SalePrice)")
plt.show()

# Apply log transformation to SalePrice to handle skewness
data['SalePrice'] = np.log1p(data['SalePrice'])

# Separate features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Select numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# ------------------------------
# Step 3: Data Transformation
# ------------------------------

# Preprocessing for numerical data
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# ------------------------------
# Step 4: Model Training with Hyperparameter Tuning
# ------------------------------

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=10000),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Hyperparameter grids
param_grids = {
    "Ridge Regression": {"model__alpha": [0.1, 1, 10, 100]},
    "Lasso Regression": {"model__alpha": [0.01, 0.1, 1, 10]},
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
}

# Create a pipeline for each model
pipelines = {
    name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    for name, model in models.items()
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
results = {}
best_models = {}

for name, pipeline in pipelines.items():
    if name in param_grids:
        # Perform hyperparameter tuning using GridSearchCV
        grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model
        print(f"\n{name} Best Parameters: {grid.best_params_}")
    else:
        # Train model without hyperparameter tuning
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_models[name] = best_model

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "MAE": mae, "R^2": r2}
    print(f"\n{name} Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R^2): {r2:.2f}")

# ------------------------------
# Step 5: Comparison of Models
# ------------------------------

# Create a comparison dataframe
comparison_df = pd.DataFrame(results).T

# Display comparison
print("\nModel Comparison:")
print(comparison_df)

# Plot R-squared values for all models
plt.figure(figsize=(10, 6))
sns.barplot(x=comparison_df.index, y=comparison_df["R^2"])
plt.title("R-squared Comparison Across Models")
plt.ylabel("R-squared")
plt.xlabel("Regression Model")
plt.show()
