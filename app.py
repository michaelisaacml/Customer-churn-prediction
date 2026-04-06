import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# sample training data
data = {
    "tenure": [1, 34, 2, 45, 2, 8, 22, 10],
    "monthly_charges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75],
    "total_charges": [29.85, 1889.5, 108.15, 1840.75, 151.65, 820.5, 1949.4, 301.9],
    "churn": [1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("churn", axis=1)
y = df["churn"]

model = RandomForestClassifier()
model.fit(X, y)

st.title("Customer Churn Prediction")

tenure = st.number_input("Customer Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        [[tenure, monthly, total]],
        columns=["tenure", "monthly_charges", "total_charges"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
