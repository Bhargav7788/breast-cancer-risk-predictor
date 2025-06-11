import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- Page Setup ----------
st.set_page_config(page_title="Breast Cancer Risk Predictor", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9fb;
    }
    .stButton > button {
        font-size: 18px;
        padding: 0.5em 2em;
        border-radius: 12px;
    }
    .css-1cpxqw2 {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load and Prepare Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test, scaler, df.columns[1:]

X, y, X_train, X_test, y_train, y_test, scaler, feature_names = load_data()

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------- UI ----------
st.title("🩺 Breast Cancer Risk Predictor")
st.write("Enter the values below to predict if a tumor is **malignant** or **benign**.")

user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", min_value=0.0, value=0.0, format="%.4f")
    user_input.append(val)

# ---------- Prediction ----------
if st.button("🔍 Predict"):
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧠 Prediction Result")
    if prediction == 1:
        st.error(f"**Malignant Tumor Detected** 🚨\n\nRisk Score: `{proba:.2f}`")
    else:
        st.success(f"**Benign Tumor Detected** ✅\n\nRisk Score: `{proba:.2f}`")

    # ---------- SHAP Explainability ----------
    st.subheader("🔎 Feature Importance")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_scaled)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
