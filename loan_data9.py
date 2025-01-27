import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from collections import Counter
import joblib

# Load Dataset
data = pd.read_csv(r"C:\Users\NAVYA\Downloads\loan_approval_dataset.csv")
data.columns = data.columns.str.strip()  # Clean column names

# Handle Missing Values
numerical_features = data.select_dtypes(include=["float64", "int64"]).columns
categorical_features = data.select_dtypes(include=["object"]).columns

for col in numerical_features:
    data[col].fillna(data[col].median(), inplace=True)

for col in categorical_features:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode Categorical Variables
label_encoders = {}
for col in categorical_features:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Feature Engineering
data["debt_to_income_ratio"] = data["loan_amount"] / (data["income_annum"] + 1)
data["asset_to_loan_ratio"] = (
    data["residential_assets_value"] + data["commercial_assets_value"] + data["luxury_assets_value"]
) / (data["loan_amount"] + 1)
data["asset_to_income_ratio"] = (
    data["residential_assets_value"] + data["commercial_assets_value"] + data["luxury_assets_value"]
) / (data["income_annum"] + 1)

# Replace infinite values and drop NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Prepare Features and Target
X = data.drop(columns=["loan_status"], errors="ignore")  # Drop target column
y = data["loan_status"]

# Save feature names for consistency
feature_names = X.columns.tolist()

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Class Imbalance with SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_resampled_scaled, y_resampled)

best_rf = grid_search.best_estimator_


y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Print Metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Accuracy Score:", accuracy_score(y_test, y_pred))



# Save Model, Scaler, and Feature Names
joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


