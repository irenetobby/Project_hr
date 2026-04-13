import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# Load assets
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
metrics = joblib.load('metrics.pkl')

# Sidebar
st.sidebar.title("Model Performance")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
st.sidebar.metric("R² Score", f"{metrics['r2']:.4f}")
st.sidebar.metric("Avg. Error (MAE)", f"${metrics['mae']:,.0f}")

# Main UI
st.title("💰 Salary Predictor (Log-Scaled)")

dept_list = preprocessor.named_transformers_['ohe_dept'].categories_[0]
level_list = preprocessor.named_transformers_['oe_joblevel'].categories_[0]

dept = st.selectbox("Department", dept_list)
level = st.selectbox("Job Level", level_list)

if st.button("Calculate Salary", type="primary"):
    input_df = pd.DataFrame({'Department': [dept], 'Job_Level': [level]})
    X_input = preprocessor.transform(input_df)
    
    # 1. Get Log Prediction
    log_pred = model.predict(X_input)[0]
    
    # 2. Convert back to Dollars
    final_salary = np.exp(log_pred)
    
    st.markdown("---")
    st.subheader("Estimated Salary:")
    st.header(f":green[${final_salary:,.2f}]")
