
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained pipeline
# This pipeline includes: preprocessor and the classifier
try:
    pipeline = joblib.load('best_churn_model_pipeline.pkl')
    st.success("Model pipeline loaded successfully! 🤖")
except FileNotFoundError:
    st.error("Model file 'best_churn_model_pipeline.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set Streamlit page title
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📈")
st.title('Customer Churn Prediction 📈')
st.markdown(f'Enter customer details below to predict the likelihood of churn. This app uses a **Random Forest** model.')

# Create the input form
with st.form(key='churn_form'):
    st.header('Customer Details')
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input('Age', min_value=18, max_value=100, value=30, help="Customer's age (18-100).")
        Tenure = st.number_input('Tenure (months)', min_value=1, max_value=72, value=12, help="How many months the customer has been with the company.")
        Usage_Frequency = st.number_input('Usage Frequency (per month)', min_value=1, max_value=30, value=15, help="Number of times the service is used per month.")
        Support_Calls = st.number_input('Support Calls', min_value=0, max_value=10, value=2, help="Number of support calls made by the customer.")
        Gender = st.selectbox('Gender', options=['Female', 'Male'], help="Customer's gender.")

    with col2:
        Payment_Delay = st.number_input('Payment Delay (days)', min_value=0, max_value=30, value=5, help="Average delay in payment in days.")
        Total_Spend = st.number_input('Total Spend ($)', min_value=100.0, max_value=1000.0, value=500.0, format="%.2f", help="Total amount spent by the customer.")
        Last_Interaction = st.number_input('Last Interaction (days ago)', min_value=1, max_value=30, value=10, help="Days since the customer's last interaction.")
        Subscription_Type = st.selectbox('Subscription Type', options=['Standard', 'Basic', 'Premium'], help="Type of subscription.")
        Contract_Length = st.selectbox('Contract Length', options=['Annual', 'Monthly', 'Quarterly'], help="Length of the customer's contract.")
    
    # Submit button (CORRECTED)
    submit_button = st.form_submit_button(label='Predict Churn')

# When the button is pressed
if submit_button:
    # 1. Create a DataFrame from the inputs
    # The column names MUST match those used during training
    input_data = pd.DataFrame(
        [[
            Age, Tenure, Usage_Frequency, Support_Calls, Payment_Delay, 
            Total_Spend, Last_Interaction, Gender, Subscription_Type, Contract_Length
        ]],
        columns=['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender', 'Subscription Type', 'Contract Length']
    )
    
    st.write("---")
    st.subheader("Input Data:")
    st.dataframe(input_data)
    
    # 2. Make prediction
    # The pipeline will handle all preprocessing
    try:
        # Note: The 'pipeline' object already contains the trained model
        # We call .predict() directly on the pipeline
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        
        # 3. Display the result
        st.subheader('Prediction Result')
        
        if prediction == 1:
            churn_prob = prediction_proba[1] * 100
            st.error(f'**Prediction: Customer is likely to CHURN** (Confidence: {churn_prob:.2f}%)')
            st.warning("Consider taking retention actions for this customer.")
        else:
            stay_prob = prediction_proba[0] * 100
            st.success(f'**Prediction: Customer is likely to STAY** (Confidence: {stay_prob:.2f}%)')
            st.info("This customer appears loyal.")
            
        st.write("---")
        st.subheader("Prediction Probabilities")
        
        prob_df = pd.DataFrame({
            'Class': ['Stay (0)', 'Churn (1)'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        
        st.bar_chart(prob_df.set_index('Class'))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
