import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler

# Suppress FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Load data
df = pd.read_csv('insurance.csv')

# Encode categorical variables
df['sex'] = df['sex'].replace({'male': 0, 'female': 1}).astype(np.int8)
df['smoker'] = df['smoker'].replace({'yes': 0, 'no': 1}).astype(np.int8)
df['region'] = df['region'].replace({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}).astype(np.int8)

# Define features and target variable
X = df.drop(columns='charges', axis=1)
Y = df['charges']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=10000,
        learning_rate_init=0.0005,
        random_state=2,
        verbose=False,
        early_stopping=True,
        n_iter_no_change=100
    )
}

# Stacking model
base_models = [
    ('linear', models['Linear Regression']), 
    ('ridge', models['Ridge Regression']), 
    ('nn', models['Neural Network'])
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Train models only if not already trained
if 'models' not in st.session_state:
    st.session_state.models = {name: model.fit(X_train, Y_train) for name, model in {**models, 'Stacking Model': stacking_model}.items()}

# Streamlit interface
st.title('Medical Cost Prediction')

# User input
age = st.number_input('Age:', min_value=0)
sex = st.selectbox('Gender:', ('male', 'female'))
bmi = st.number_input('BMI:', min_value=0.0)
children = st.number_input('Number of Children:', min_value=0)
smoker = st.selectbox('Smoker?', ('yes', 'no'))
region = st.selectbox('Region:', ('southeast', 'southwest', 'northeast', 'northwest'))

# Convert user input into the appropriate format
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'female' else 0],  # Adjust encoding based on user input
    'bmi': [bmi],
    'children': [children],
    'smoker': [0 if smoker == 'yes' else 1],  # Adjust encoding based on user input
    'region': [1 if region == 'southwest' else 0 if region == 'southeast' else 2 if region == 'northeast' else 3]  # Adjust encoding
})

# Ensure input_data has the same column names and order as X
input_data = input_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Select model
model_choice = st.selectbox('Choose a model for prediction:', list(st.session_state.models.keys()))

# Predict and display results when button is clicked
if st.button('Predict'):
    model = st.session_state.models[model_choice]
    prediction = model.predict(input_data_scaled)
    st.subheader('Prediction Result:')
    st.write(f'{model_choice}: {prediction[0]:.2f} USD')
