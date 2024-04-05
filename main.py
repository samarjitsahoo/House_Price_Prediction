import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('Dragon.joblib')

# Define the input fields for user input
st.title('Selmon Real Estate - Price Predictor')
st.write('Enter the following details to get the predicted house price:')

CRIM = st.number_input('CRIM (per capita crime rate by town)')
ZN = st.number_input('ZN (proportion of residential land zoned for lots over 25,000 sq.ft.)')
INDUS = st.number_input('INDUS (proportion of non-retail business acres per town)')
CHAS = st.number_input('CHAS (Charles River dummy variable)')
NOX = st.number_input('NOX (nitric oxides concentration)')
RM = st.number_input('RM (average number of rooms per dwelling)')
AGE = st.number_input('AGE (proportion of owner-occupied units built prior to 1940)')
DIS = st.number_input('DIS (weighted distances to five Boston employment centres)')
RAD = st.number_input('RAD (index of accessibility to radial highways)')
TAX = st.number_input('TAX (full-value property-tax rate per $10,000)')
PTRATIO = st.number_input('PTRATIO (pupil-teacher ratio by town)')
B = st.number_input('B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)')
LSTAT = st.number_input('LSTAT (% lower status of the population)')

# Create a dictionary with the user inputs
input_data = {
    'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS,
    'NOX': NOX, 'RM': RM, 'AGE': AGE, 'DIS': DIS,
    'RAD': RAD, 'TAX': TAX, 'PTRATIO': PTRATIO, 'B': B,
    'LSTAT': LSTAT
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions using the model
prediction = model.predict(input_df)

# Display the predicted price
st.subheader('Predicted House Price')
st.write(f'The predicted house price is ${prediction[0] * 1000:.2f}')
