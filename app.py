import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================
# Load dataset
# ============================
df = pd.read_csv('employeeproductivitydataset.csv')

# ============================
# Encode categorical variables
# ============================
df['Department'] = df['Department'].map({
    'Sales': 0,
    'HR': 1,
    'IT': 2,
    'Finance': 3
})

df['ManagerFeedback'] = df['ManagerFeedback'].map({
    'Poor': 0,
    'Average': 1,
    'Good': 2,
    'Excellent': 3
})

# Drop unnecessary columns
df = df.drop(['EmployeeID', 'EmployeeName'], axis=1, errors='ignore')

# ============================
# Train Model
# ============================
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']   # 1–5 rating

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
def get_feature_importance(input_data):
    importance = model.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    return imp_df

def explain_underperformance(input_data):
    issues = []

    if input_data['KPI_Score'].values[0] < 50:
        issues.append("Low KPI score")

    if input_data['Attendance'].values[0] < 70:
        issues.append("Poor attendance")

    if input_data['Productivity'].values[0] < 50:
        issues.append("Low productivity")

    if input_data['AbsenteeismRate'].values[0] > 10:
        issues.append("High absenteeism")

    if input_data['ProjectsCompleted'].values[0] < 3:
        issues.append("Few completed projects")

    if len(issues) == 0:
        issues.append("No major issues detected")

    return issues

# ============================
# UI
# ============================
st.title("Employee Performance Predictor")

st.header("Enter Employee Data")

# Inputs
kpi = st.slider("KPI Score", 0, 100, 60)
attendance = st.slider("Attendance (%)", 0, 100, 80)
training = st.slider("Trainings Attended", 0, 20, 5)
productivity = st.slider("Productivity Score", 0, 100, 65)
absenteeism = st.slider("Absenteeism Rate (%)", 0, 30, 5)
turnover = st.slider("Turnover Rate (%)", 0, 30, 5)
projects = st.slider("Projects Completed", 0, 20, 5)

department = st.selectbox("Department", ['Sales', 'HR', 'IT', 'Finance'])
department = {'Sales': 0, 'HR': 1, 'IT': 2, 'Finance': 3}[department]

manager_feedback = st.selectbox(
    "Manager Feedback", ['Poor', 'Average', 'Good', 'Excellent']
)
manager_feedback = {
    'Poor': 0,
    'Average': 1,
    'Good': 2,
    'Excellent': 3
}[manager_feedback]

# Create input dataframe
input_data = pd.DataFrame({
    'KPI_Score': [kpi],
    'Attendance': [attendance],
    'TrainingCount': [training],
    'Productivity': [productivity],
    'AbsenteeismRate': [absenteeism],
    'TurnoverRate': [turnover],
    'ProjectsCompleted': [projects],
    'Department': [department],
    'ManagerFeedback': [manager_feedback]
})

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# ============================
# Prediction
# ============================
if st.button("Predict Performance"):

    prediction, probability = predict_performance(input_data)
    rating = prediction[0]

    st.success(f"Predicted Performance Rating: {rating} / 5")

    # ============================
    # Feature Importance
    # ============================
    st.subheader("Key Factors Influencing Performance")

    importance_df = get_feature_importance(input_data)

    st.bar_chart(importance_df.set_index('Feature'))

    # ============================
    # Explanation
    # ============================
    st.subheader("Why this rating?")

    issues = explain_underperformance(input_data)

    for issue in issues:
        st.write(f"- {issue}")

    # ============================
    # Recommendation
    # ============================
    st.subheader("HR Recommendation")

    if rating <= 2:
        st.error("High Risk: Immediate improvement plan required")
    elif rating == 3:
        st.warning("Average Performance: Provide training & mentorship")
    else:
        st.success("High Performer: Consider promotion or rewards")
