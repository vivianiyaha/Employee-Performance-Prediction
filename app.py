import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# Load Dataset
# ============================
df = pd.read_csv('employee_performance.csv')

# ============================
# Clean Data
# ============================
df.columns = df.columns.str.strip()

# Drop useless columns
df = df.drop(['EmployeeName', 'EmployeeID'], axis=1)

# ============================
# Create Target (PerformanceRating)
# ============================
def rate_performance(row):
    score = (
        row['KPIScore'] * 0.4 +
        row['ProductivityScore'] * 0.4 +
        row['Attendance'] * 0.2
    )

    if score >= 85:
        return 5
    elif score >= 75:
        return 4
    elif score >= 65:
        return 3
    elif score >= 55:
        return 2
    else:
        return 1

df['PerformanceRating'] = df.apply(rate_performance, axis=1)

# ============================
# Encode Department
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
# Train Model
# ============================
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ============================
# Functions
# ============================
def predict_performance(data):
    prediction = model.predict(data)
    probs = model.predict_proba(data)
    confidence = np.max(probs)
    return prediction[0], confidence


def explain_prediction(input_df):
    importance = model.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    return imp_df.head(5)


def performance_label(score):
    return {
        5: "⭐ Excellent",
        4: "👍 Good",
        3: "🙂 Average",
        2: "⚠️ Below Average",
        1: "❌ Poor"
    }[score]


def recommendation(rating):
    if rating >= 4:
        return "Promote / Reward Employee"
    elif rating == 3:
        return "Provide Training & Monitor"
    else:
        return "Immediate Improvement Plan Needed"


# ============================
# UI
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "About"])

# ============================
# Predictor Page
# ============================
if page == "Predictor":

    st.title("📊 Employee Performance Predictor")

    st.write("Predict employee performance and understand key drivers.")

    # Inputs
    dept = st.selectbox("Department", ['IT', 'Marketing', 'HR', 'Finance', 'Sales', 'Operations'])
    dept = {'IT': 0, 'Marketing': 1, 'HR': 2, 'Finance': 3, 'Sales': 4, 'Operations': 5}[dept]

    kpi = st.slider("KPI Score", 0, 100, 70)
    attendance = st.slider("Attendance", 0, 100, 80)
    training = st.slider("Number of Training", 0, 10, 3)
    productivity = st.slider("Productivity Score", 0, 100, 75)
    absenteeism = st.slider("Absenteeism Rate", 0, 10, 2)
    turnover = st.slider("Turnover Rate", 0, 30, 10)
    feedback = st.slider("Manager Feedback (1-5)", 1, 5, 3)
    projects = st.slider("Projects Completed", 0, 20, 5)

    # Create input dataframe
    input_df = pd.DataFrame({
        'Department': [dept],
        'KPIScore': [kpi],
        'Attendance': [attendance],
        'NumberofTraining': [training],
        'ProductivityScore': [productivity],
        'AbsenteeismRate': [absenteeism],
        'TurnoverRate': [turnover],
        'ManagerFeedback': [feedback],
        'ProjectsCompleted': [projects]
    })

    # Align columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Prediction
    if st.button("Predict Performance"):

        pred, confidence = predict_performance(input_df)
        label = performance_label(pred)
        advice = recommendation(pred)

        st.success(f"Predicted Rating: {pred} ({label})")
        st.write(f"Confidence: {confidence:.2%}")

        st.write("### 📌 Recommendation:")
        st.info(advice)

        # Feature importance
        st.write("### 🔍 Why this prediction?")
        importance_df = explain_prediction(input_df)

        st.bar_chart(importance_df.set_index('Feature'))

# ============================
# About Page
# ============================
elif page == "About":

    st.title("About Project")

    st.write("""
    ### 🎯 Problem
    Organizations struggle to fairly evaluate employee performance.

    ### 💡 Solution
    This ML model predicts employee performance using objective metrics.

    ### 🤖 Model
    Random Forest Classifier

    ### 📊 Output
    Performance rating (1–5) with explanation.

    ### 👩‍💻 Built By
    Vivian Iyaha
    """)
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
