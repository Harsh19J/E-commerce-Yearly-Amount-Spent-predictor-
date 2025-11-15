import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import os

MODEL_FILENAME = 'linear_regression_model.pkl'

with open(MODEL_FILENAME, 'rb') as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Yearly Spending Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🛍️ E-commerce Yearly Spending Predictor")
st.markdown("""
    Adjust the input parameters below to see the predicted **Yearly Amount Spent**
    (in dollars) based on the linear regression model.
""")

st.subheader("Customer Activity Metrics")

col1, col2 = st.columns(2)

with col1:
    avg_session = st.slider(
        'Avg. Session Length (minutes)',
        min_value=29.0, max_value=37.0, value=33.0, step=0.1, format="%.2f"
    )

with col2:
    time_on_app = st.slider(
        'Time on App (minutes)',
        min_value=8.0, max_value=16.0, value=12.0, step=0.1, format="%.2f"
    )

col3, col4 = st.columns(2)

with col3:
    time_on_website = st.slider(
        'Time on Website (minutes)',
        min_value=33.0, max_value=41.0, value=37.0, step=0.1, format="%.2f"
    )

with col4:
    length_membership = st.slider(
        'Length of Membership (Years)',
        min_value=0.0, max_value=7.0, value=3.5, step=0.1, format="%.2f"
    )

features = np.array([
    [avg_session, time_on_app, time_on_website, length_membership]
])

predicted_amount = model.predict(features)[0]

if predicted_amount < 0:
    predicted_amount = 0.0


st.markdown("---")

st.subheader("🎯 Predicted Yearly Amount Spent")
st.success(f"**${predicted_amount:,.2f}**")

st.markdown("""
    *Interpretation: This is the estimated yearly spending for a customer with the selected activity levels.*
""")

st.subheader("Model Feature Coefficients")
st.info("The coefficients show the impact of each variable on the Yearly Amount Spent.")

feature_names = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
coefficients = model.coef_

coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

coeff_df['Absolute Value'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Absolute Value', ascending=False).drop(columns=['Absolute Value'])

st.dataframe(coeff_df, use_container_width=True, hide_index=True)