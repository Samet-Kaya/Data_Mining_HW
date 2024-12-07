# Importing necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------
# Step 1: Load the Dataset
# ------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
os.chdir(script_dir)  # Change working directory to script's location

df = pd.read_csv('titanic.csv')  # Adjust file path if needed

# Display the first few rows of the dataset
print("First five rows of the dataset:")
print(df.head())

# Display basic statistics of the dataset
print("\nDataset statistics:")
print(df.describe())

# Display the class distribution
print("\nClass distribution (Survived):")
print(df['Survived'].value_counts())

# ------------------------------
# Step 2: Data Visualization
# ------------------------------
# Class distribution
sns.countplot(x='Survived', data=df, palette=['red', 'green'], hue='Survived')
plt.title("Survival Count")
plt.show()

# Survival based on gender
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')
plt.title("Survival Based on Gender")
plt.show()

# Survival based on passenger class
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set3')
plt.title("Survival Based on Passenger Class")
plt.show()

# Survival based on age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', palette=['red', 'green'])
plt.title("Survival Based on Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["Did not survive", "Survived"], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------
# Step 3: Data Preprocessing
# ------------------------------
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

# ------------------------------
# Step 4: Split the Data (Train, Validation, Test)
# ------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ------------------------------
# Step 5: Model Training and Evaluation
# ------------------------------
def evaluate_model(model, X_val, y_val, X_test, y_test, model_name):
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    print(f"\n{model_name} Validation Performance:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.2f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.2f}")
    print(f"F1 Score: {f1_score(y_val, y_val_pred):.2f}")
    print(f"\n{model_name} Test Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.2f}")

# ------------------------------
# Hyperparameter Tuning
# ------------------------------
def tune_hyperparameters(model, param_grid, X_train, y_train, model_name):
    search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")
    return search.best_estimator_

# Decision Tree
dt_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30]}
dt_model = tune_hyperparameters(DecisionTreeClassifier(random_state=42), dt_param_grid, X_train, y_train, "Decision Tree")
evaluate_model(dt_model, X_val, y_val, X_test, y_test, "Tuned Decision Tree")

# Random Forest
rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
rf_model = tune_hyperparameters(RandomForestClassifier(random_state=42), rf_param_grid, X_train, y_train, "Random Forest")
evaluate_model(rf_model, X_val, y_val, X_test, y_test, "Tuned Random Forest")

# SVM
svm_param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale']}
svm_model = tune_hyperparameters(SVC(random_state=42), svm_param_grid, X_train, y_train, "Support Vector Machine")
evaluate_model(svm_model, X_val, y_val, X_test, y_test, "Tuned Support Vector Machine")

# ------------------------------
# Feature Importance and Confusion Matrices
# ------------------------------
# Feature importance for Random Forest
importances_rf = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_rf, y=features)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Feature importance for Decision Tree
importances_dt = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_dt, y=features)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Confusion matrices
models = {
    "Decision Tree": dt_model.predict(X_test),
    "Random Forest": rf_model.predict(X_test),
    "SVM": svm_model.predict(X_test)
}

for model_name, y_pred in models.items():
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"{model_name} Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
