import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.set_page_config(page_title="Fuel Efficiency Predictor", page_icon="🚗")

st.title("🚗 Fuel Efficiency Predictor")
st.markdown("Predict how many **miles per gallon (MPG)** a car will achieve based on its specifications.")

st.sidebar.header("Enter Car Specifications")

cylinders = st.sidebar.selectbox("Cylinders", [3, 4, 5, 6, 8])
displacement = st.sidebar.slider("Displacement (cu. inches)", 68, 455, 200)
horsepower = st.sidebar.slider("Horsepower", 46, 230, 100)
weight = st.sidebar.slider("Weight (lbs)", 1613, 5140, 2800)
acceleration = st.sidebar.slider("Acceleration (0-60 mph in sec)", 8.0, 24.8, 15.0)
model_year = st.sidebar.slider("Model Year (e.g. 70 = 1970)", 70, 82, 76)
origin = st.sidebar.selectbox("Origin", [1, 2, 3], format_func=lambda x: {1: "American", 2: "European", 3: "Japanese"}[x])

input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])

prediction = model.predict(input_data)[0]

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted MPG", f"{prediction:.1f} MPG")
col2.metric("R² Score", "0.79")
col3.metric("RMSE", "3.27")

if prediction >= 30:
    st.success(f"✅ Great fuel efficiency! This car achieves {prediction:.1f} MPG.")
elif prediction >= 20:
    st.warning(f"⚠️ Average fuel efficiency. This car achieves {prediction:.1f} MPG.")
else:
    st.error(f"❌ Poor fuel efficiency. This car achieves {prediction:.1f} MPG.")

st.markdown("---")
st.subheader("Feature Importance")
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
coefficients = model.coef_

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['red' if c < 0 else 'green' for c in coefficients]
ax.barh(features, coefficients, color=colors)
ax.set_xlabel("Coefficient Value")
ax.set_title("How Each Feature Affects MPG")
ax.axvline(x=0, color='black', linewidth=0.8)
st.pyplot(fig)

st.markdown("---")
st.subheader("Actual vs Predicted MPG")
df = pd.read_csv("auto-mpg-clean.csv")
X = df.drop(columns=['mpg'])
y_actual = df['mpg']
y_pred = model.predict(X)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(y_actual, y_pred, alpha=0.5, color='steelblue')
ax2.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
ax2.set_xlabel("Actual MPG")
ax2.set_ylabel("Predicted MPG")
ax2.set_title("Actual vs Predicted MPG")
st.pyplot(fig2)