import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# Load Dataset
# ============================
df = pd.read_csv('employeeproductivitydatasete.csv')

# ============================
# Clean Data
# ============================
df.columns = df.columns.str.strip()
df = df.drop(['EmployeeName', 'EmployeeID'], axis=1)

# ============================
# Create Target
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
# FUNCTIONS (ALL LOGIC HERE)
# ============================

def predict_performance(data):
    prediction = model.predict(data)
    probs = model.predict_proba(data)
    confidence = np.max(probs)
    return prediction[0], confidence


def explain_prediction():
    importance = model.feature_importances_
    feature_names = X.columns

    return pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)


def get_issues(input_data):
    issues = []

    if input_data['KPIScore'].values[0] < 60:
        issues.append("Low KPI Score")

    if input_data['Attendance'].values[0] < 70:
        issues.append("Poor Attendance")

    if input_data['ProductivityScore'].values[0] < 60:
        issues.append("Low Productivity")

    if input_data['ProjectsCompleted'].values[0] < 5:
        issues.append("Low Project Output")

    if input_data['ManagerFeedback'].values[0] <= 2:
        issues.append("Poor Manager Feedback")

    return issues


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
# STREAMLIT UI
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "About"])

# ============================
# Predictor Page
# ============================
if page == "Predictor":

    st.title("📊 Employee Performance Predictor")

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

    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Prediction
    if st.button("Predict Performance"):

        pred, confidence = predict_performance(input_df)

        st.success(f"Rating: {pred} ({performance_label(pred)})")
        st.write(f"Confidence: {confidence:.2%}")

        # Recommendation
        st.subheader("📌 Recommendation")
        st.info(recommendation(pred))

        # Feature Importance
        st.subheader("🔍 Key Drivers")
        importance_df = explain_prediction()
        st.bar_chart(importance_df.set_index('Feature'))

        # Issues
        st.subheader("⚠️ Performance Issues")
        issues = get_issues(input_df)

        if issues:
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("No major issues detected")

# ============================
# About Page
# ============================
elif page == "About":

    st.title("About Project")

    st.write("""
    ### 🎯 Problem
    Organizations struggle to fairly evaluate employee performance.

    ### 💡 Solution
    Predict performance using data-driven insights.

    ### 🤖 Model
    Random Forest Classifier

    ### 📊 Output
    Performance rating (1–5)
    """)

elif page == "Profile":
    
    st.title("Profile")
    
    st.write("""
    Vivian Iyaha is a University of Port Harcourt graduate with a degree in Management. She is skilled in management and has a strong interest in emerging technologies like Machine Learning and Artificial Intelligence.
    Vivian is seeking opportunities to apply her HR expertise and learn more about these technologies. She is enthusiastic about collaborating on projects that leverage Machine Learning and Artificial Intelligence to solve real-world problems and drive innovation.
    Connect with Vivian on [LinkedIn](https://www.linkedin.com/in/vivian-i-556499126/)
    """)

    
