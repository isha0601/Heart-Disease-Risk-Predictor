# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---- Page Config ----
st.set_page_config(page_title="❤️ Heart Disease Risk Predictor")

st.title("❤️ Heart Disease Risk Predictor")
st.write("**Educational tool to predict heart disease risk.** _Not for clinical use!_")

# ---- Load Dataset ----
@st.cache_data
def load_data():
    local_file = "heart_cleveland_upload.csv"

    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
    else:
        st.error(f"Could not find `{local_file}`. Please make sure it’s in your project folder.")
        st.stop()

    return df

df = load_data()

# ---- EDA ----
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df.head())

if st.checkbox("Show correlation heatmap"):
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), ax=ax, annot=True, cmap="coolwarm")
    st.pyplot(fig)

# ---- Prepare Data ----
# In your CSV the target column might have different name.
# Check column names:
st.write("Columns:", df.columns.tolist())

# Adjust this if needed:
target_col = "target" if "target" in df.columns else df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train Model ----
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.success(f"✅ **Model Accuracy:** {accuracy:.2f}")

# ---- User Input ----
st.header("📋 Enter your health details")

def user_input_features():
    age = st.slider("Age", int(X.age.min()), int(X.age.max()), int(X.age.mean()))
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", int(X.trestbps.min()), int(X.trestbps.max()), int(X.trestbps.mean()))
    chol = st.slider("Serum Cholestoral (mg/dl)", int(X.chol.min()), int(X.chol.max()), int(X.chol.mean()))
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.slider("Max Heart Rate Achieved", int(X.thalach.min()), int(X.thalach.max()), int(X.thalach.mean()))
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST depression induced by exercise", float(X.oldpeak.min()), float(X.oldpeak.max()), float(X.oldpeak.mean()))
    slope = st.slider("Slope of peak exercise ST segment (0-2)", 0, 2, 1)
    ca = st.slider("Major vessels colored by flourosopy (0-4)", 0, 4, 0)
    thal = st.slider("Thal (0-3)", 0, 3, 2)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# ---- Prediction ----
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

st.subheader("🔍 Prediction")
result = "✅ **Low Risk of Heart Disease**" if prediction[0] == 0 else "⚠️ **High Risk of Heart Disease**"
st.write(result)

st.write(f"**Probability of Prediction:** {prediction_proba[0][prediction][0]:.2f}")

st.write("---")
st.caption("🚑 *This tool is for learning only — not medical advice!*")
