import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# Load dataset
# ============================
df = pd.read_csv('employeeproductivitydatasete.csv')

# Clean column names
df.columns = df.columns.str.strip()

# ============================
# Encode categorical variables
# ============================
df['Department'] = df['Department'].map({
    'IT': 0,
    'Marketing': 1,
    'HR': 2,
    'Finance': 3,
    'Sales': 4,
    'Operations': 5
})

# ============================
# Drop unnecessary columns
# ============================
df = df.drop(['EmployeeName', 'EmployeeID'], axis=1)

# ============================
# Features & Target
# ============================
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']

# ============================
# Train Model
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ============================
# Prediction Function
# ============================
def predict_performance(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    return prediction, probability

# ============================
# Explainability Function
# ============================
def explain_underperformance(input_data):
    issues = []

    if input_data['KPIScore'].values[0] < 60:
        issues.append("Low KPI Score")

    if input_data['Attendance'].values[0] < 75:
        issues.append("Low Attendance")

    if input_data['ProductivityScore'].values[0] < 60:
        issues.append("Low Productivity")

    if input_data['AbsenteeismRate'].values[0] > 7:
        issues.append("High Absenteeism")

    if input_data['ProjectsCompleted'].values[0] < 5:
        issues.append("Low Project Output")

    if input_data['ManagerFeedback'].values[0] <= 2:
        issues.append("Poor Manager Feedback")

    return issues

# ============================
# STREAMLIT UI
# ============================
st.title("Employee Performance Predictor")

st.header("Enter Employee Data")

# Inputs
kpi = st.slider("KPI Score", 0, 100, 60)
attendance = st.slider("Attendance (%)", 0, 100, 80)
training = st.slider("Number of Trainings", 0, 10, 3)
productivity = st.slider("Productivity Score", 0, 100, 65)
absenteeism = st.slider("Absenteeism Rate (%)", 0, 10, 3)
turnover = st.slider("Turnover Rate (%)", 0, 30, 10)
projects = st.slider("Projects Completed", 0, 20, 5)

department = st.selectbox(
    "Department",
    ['IT', 'Marketing', 'HR', 'Finance', 'Sales', 'Operations']
)

department = {
    'IT': 0,
    'Marketing': 1,
    'HR': 2,
    'Finance': 3,
    'Sales': 4,
    'Operations': 5
}[department]

manager_feedback = st.slider("Manager Feedback (1-5)", 1, 5, 3)

# ============================
# Create Input Data
# ============================
input_data = pd.DataFrame({
    'KPIScore': [kpi],
    'Attendance': [attendance],
    'NumberofTraining': [training],
    'ProductivityScore': [productivity],
    'AbsenteeismRate': [absenteeism],
    'TurnoverRate': [turnover],
    'ManagerFeedback': [manager_feedback],
    'ProjectsCompleted': [projects],
    'Department': [department]
})

# Align with training columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# ============================
# Prediction Button
# ============================
if st.button("Predict Performance"):

    prediction, probability = predict_performance(input_data)
    rating = prediction[0]

    st.success(f"Predicted Performance Rating: {rating} / 5")

    # ============================
    # Feature Importance
    # ============================
    st.subheader("Key Drivers of Performance")

    importance = model.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(imp_df.set_index('Feature'))

    # ============================
    # Explanation
    # ============================
    st.subheader("Why this rating?")

    issues = explain_underperformance(input_data)

    if len(issues) == 0:
        st.success("No major performance issues detected")
    else:
        for issue in issues:
            st.write(f"- {issue}")

    # ============================
    # Recommendation
    # ============================
    st.subheader("HR Recommendation")

    if rating <= 2:
        st.error("Immediate intervention required")
    elif rating == 3:
        st.warning("Needs improvement & monitoring")
    else:
        st.success("High performer — reward or promote")
