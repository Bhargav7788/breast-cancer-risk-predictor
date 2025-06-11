import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Step 3: Encode the target column
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Step 4: Split features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train models
log_reg = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(probability=True)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Step 8: Evaluate models
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "Support Vector Machine": svm,
    "XGBoost": xgb
}

print("📊 Model Evaluation Results:\n")
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"🔹 {name}")
    print(f"   ✅ Accuracy:   {accuracy_score(y_test, y_pred):.4f}")
    print(f"   ✅ F1 Score:   {f1_score(y_test, y_pred):.4f}")
    print(f"   ✅ ROC AUC:    {roc_auc_score(y_test, y_proba):.4f}")
    print("-" * 40)

# Step 9: SHAP Explainability for Logistic Regression
print("\n🔍 SHAP Feature Importance (Logistic Regression)...")

# Create SHAP explainer
explainer = shap.Explainer(log_reg, X_train)
shap_values = explainer(X_test)

# Show SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.show()