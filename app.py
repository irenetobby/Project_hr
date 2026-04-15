import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# Load assets
try:
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    metrics = joblib.load('metrics.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run your training script first!")
    st.stop()

# Sidebar
st.sidebar.title("Model Performance")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
st.sidebar.metric("R² Score", f"{metrics['r2']:.4f}")
st.sidebar.metric("Avg. Error (MAE)", f"${metrics['mae']:,.0f}")

# Main UI
st.title("💰 Salary Predictor")

# Extract categories dynamically from the preprocessor
dept_list = preprocessor.named_transformers_['ohe_dept'].categories_[0]
level_list = preprocessor.named_transformers_['oe_joblevel'].categories_[0]

# --- UI Layout ---
dept = st.selectbox("Department", dept_list)
level = st.selectbox("Job Level", level_list)
rating = st.slider("Performance Rating", 1, 5, 3)
years = st.number_input("Experience Years", min_value=0, max_value=50, value=4)

# Static variables required by the preprocessor schema
title = "Specialist" 
date = pd.Timestamp.now()

if st.button("Calculate Salary", type="primary"):
    # Build DataFrame to match the schema the preprocessor was fitted on
    input_df = pd.DataFrame({
        'Department': [dept], 
        'Job_Title': [title], 
        'Job_Level': [level],
        'Performance_Rating': [rating],
        'Experience_Years': [years],
        'Hire_Date': [date]
    })
    
    try:
        # 1. Transform input using the loaded preprocessor
        X_input = preprocessor.transform(input_df)
        
        # 2. Predict raw value (REMOVED np.exp to fix $inf error)
        final_salary = model.predict(X_input)[0]
        
        st.markdown("---")
        st.subheader("Estimated Salary:")
        # Display the result formatted as currency
        st.header(f":green[${final_salary:,.2f}]")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")