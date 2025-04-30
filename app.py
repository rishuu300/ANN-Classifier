import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# artifacts_dir = 'D:\\Udemy\\Gen AI\\Projects\\ANN Classification\\artifacts'

# Load the model
artifacts_dir = os.path.join('artifacts')  # relative path
model_path = os.path.join(artifacts_dir, 'model.h5')
model = tf.keras.models.load_model(model_path) # type: ignore

# Load label encoder
label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Load one-hot encoder
onehot_encoder_path = os.path.join(artifacts_dir, 'onehot_encoder.pkl')
with open(onehot_encoder_path, 'rb') as file:
    onehot_encoder = pickle.load(file)

# Load scaler
scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)
    

# Streamlit App
st.title('Customer Churn Prediction')


# User input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is active Member', [0,1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-Hot-Encode 'Geography'
onehot_encoded = onehot_encoder.transform([[geography]])
geo_encoded = pd.DataFrame(onehot_encoded, columns = onehot_encoder.get_feature_names_out())

# Concatenate to original data
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded], axis = 1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')