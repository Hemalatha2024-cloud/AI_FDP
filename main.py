import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set page title
st.title("📊 Student Pass/Fail Prediction")

# Sample dataset
data = {
    'Hours_Studied': [2, 5, 1, 7, 3, 6, 2, 8, 4, 5],
    'Attendance_Percentage': [80, 95, 70, 98, 85, 100, 75, 90, 92, 88],
    'Passed': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  # 1 = Pass, 0 = Fail
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['Hours_Studied', 'Attendance_Percentage']]
y = df['Passed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter Student Details")
hours_studied = st.sidebar.slider("Hours Studied", min_value=0, max_value=10, value=5, step=1)
attendance_percentage = st.sidebar.slider("Attendance Percentage", min_value=0, max_value=100, value=80, step=5)

# Predict button
if st.sidebar.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({'Hours_Studied': [hours_studied], 'Attendance_Percentage': [attendance_percentage]})
    
    # Make prediction
    prediction = model.predict(input_data)
    result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"

    # Display result
    st.subheader("Prediction Result")
    st.success(f"The student is predicted to: **{result}**")

# Display sample dataset
st.subheader("📌 Sample Training Data")
st.dataframe(df)

# Model details
st.subheader("ℹ️ Model Information")
st.text("Algorithm: Logistic Regression")
st.text(f"Training Data Size: {len(X_train)}")
st.text(f"Test Data Size: {len(X_test)}")

