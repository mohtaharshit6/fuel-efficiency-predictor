# 🚗 Fuel Efficiency Predictor

A Machine Learning web app that predicts the fuel efficiency (MPG) of a car based on its specifications.

## 📌 Project Info
- **Course:** CSA2001 — Fundamentals in AI and ML
- **Syllabus Link:** CO4 — Linear Regression & Machine Learning Basics
- **Dataset:** UCI Auto MPG Dataset (398 records, 8 features)

## 🧠 ML Concepts Used
- Multiple Linear Regression
- Train-Test Split (80/20)
- Feature Correlation Analysis
- Model Evaluation — RMSE: 3.27 | R² Score: 0.79

## 📊 Features
- Interactive sliders for car specifications
- Real-time MPG prediction
- Feature importance chart
- Actual vs Predicted MPG visualization

## 🛠 Tech Stack
- Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit

## ▶️ How to Run
```bash
pip install pandas scikit-learn seaborn streamlit matplotlib
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py               # Streamlit UI
├── model.py             # ML model training
├── clean.py             # Data cleaning
├── eda.py               # Exploratory Data Analysis
├── explore.py           # Data exploration
├── auto-mpg.csv         # Original dataset
├── auto-mpg-clean.csv   # Cleaned dataset
├── model.pkl            # Saved trained model
├── heatmap.png          # Correlation heatmap
├── mpg_vs_horsepower.png
└── mpg_vs_weight.png
```

## 📈 Results
- **R² Score: 0.79** — Model explains 79% of variance in fuel efficiency
- **RMSE: 3.27** — Average prediction error of 3.27 MPG
