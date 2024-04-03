import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def accident_severity_predictor(sex, num_vehicles_involved, num_casualties):
    # Load the dataset
    # Adjust the file name accordingly
    df = pd.read_csv('RTA Dataset modified.csv')

    # Selecting relevant features
    X = df[['Sex_of_driver', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    # Label encode the 'Sex_of_driver' column
    encoder = LabelEncoder()
    X['Sex_of_driver_encoded'] = encoder.fit_transform(X['Sex_of_driver'])
    X_encoded = X[['Sex_of_driver_encoded', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    # Replace 'YourTargetVariable' with the actual target variable name in your dataset
    y = df['Accident_severity']

    # Split the dataset
    X_train, _, y_train, _ = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions using the user input
    user_input_df = pd.DataFrame({'Sex_of_driver': [sex], 'Number_of_vehicles_involved': [
                                 num_vehicles_involved], 'Number_of_casualties': [num_casualties]})

    # Label encode the user input
    user_input_df['Sex_of_driver_encoded'] = encoder.transform(user_input_df['Sex_of_driver'])
    user_input_encoded = user_input_df[['Sex_of_driver_encoded', 'Number_of_vehicles_involved', 'Number_of_casualties']]

    prediction = model.predict(user_input_encoded)

    return prediction[0]

st.title("ROAD TRAFFIC ACCIDENT SEVERITY PREDICTION")

sex_input = st.text_input("Enter the Sex (Male, Female, Unknown)")
num_vehicles_input = st.number_input("Enter the number of vehicles involved")
num_casualties_input = st.number_input("Enter the number of casualties involved")

if st.button("Predict Severity"):
    prediction = accident_severity_predictor(sex_input, num_vehicles_input, num_casualties_input)
    st.write(f"Predicted Severity: {prediction}")
