import streamlit as st
import pickle # Changed from joblib to pickle
import numpy as np
import pandas as pd
import os

# --- 1. Load the Model ---
# Ensure the 'linear_regression_model.pkl' file is in the same directory as this app.py
MODEL_FILENAME = 'linear_regression_model.pkl'

try:
    # Using pickle.load to load the model. 'rb' means read in binary mode.
    with open(MODEL_FILENAME, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Please run the 'save_model.py' script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Streamlit UI and Input Widgets ---

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

# Define reasonable default ranges based on the dataset's 'df.describe()' output:
# Avg. Session Length: mean ~33, range ~30-36
# Time on App: mean ~12, range ~8-15
# Time on Website: mean ~37, range ~34-40
# Length of Membership: mean ~3.5, range ~0.2-6.9

# Create four input widgets (sliders are great for numerical input):

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


# --- 3. Prediction Logic ---

# Prepare the input data for the model
# The order MUST match the order used during training: 
# ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']

features = np.array([
    [avg_session, time_on_app, time_on_website, length_membership]
])

# Perform the prediction
predicted_amount = model.predict(features)[0]

# Ensure the predicted amount is non-negative and format for display
if predicted_amount < 0:
    predicted_amount = 0.0

# --- 4. Display Results ---

st.markdown("---")

st.subheader("🎯 Predicted Yearly Amount Spent")
st.success(f"**${predicted_amount:,.2f}**")

st.markdown("""
    *Interpretation: This is the estimated yearly spending for a customer with the selected activity levels.*
""")

# Optional: Display the model coefficients to show feature importance
st.subheader("Model Feature Coefficients")
st.info("The coefficients show the impact of each variable on the Yearly Amount Spent.")

# Get feature names (must match training input order)
feature_names = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
coefficients = model.coef_

coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by absolute value to highlight importance
coeff_df['Absolute Value'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Absolute Value', ascending=False).drop(columns=['Absolute Value'])

st.dataframe(coeff_df, use_container_width=True, hide_index=True)