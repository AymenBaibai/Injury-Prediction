# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import base64

# Load the pre-trained model, label encoder, feature names, and accuracy
model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('feature_names.joblib')
model_accuracy = joblib.load('accuracy.joblib')

# Define the dictionary mapping injury types to prevention URLs
injury_prevention_links = {
    'Muscle strain or sprain': 'https://www.healthline.com/health/strains',
    'Joint injury (e.g., knee, shoulder)': 'https://www.healthline.com/health/joint-pain#causes',
    'Back pain or injury': 'https://www.healthline.com/health/back-pain',
    'Tendonitis or repetitive strain injury': 'https://www.healthline.com/health/tendinitis',
    'Fracture or bone injury': 'https://www.healthline.com/health/fracture#prevention'
    # Add more mappings as needed
}

# Define function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Define function to get user input
def get_user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.selectbox('Age', ['15 to 18', '19 to 25', '26 to 30', '31 to 40', '41 to 50', '51+'])
    fitness_level = st.selectbox('Current level of fitness', ['Very unfit', 'Unfit', 'Average', 'Good', 'Very good'])
    exercise_frequency = st.selectbox('How often do you exercise?', ['Never', '1 to 2 times a week', '3 to 4 times a week', 'Everyday'])
    barrier = st.selectbox('Barrier to exercising more regularly', [
        "I don't have enough time", "I don't have access to exercise facilities", 
        "I can't stay motivated", "I don't enjoy exercising", "I'll become too tired",
        "I exercise regularly with no barriers"
    ])
    exercise_type = st.selectbox('Form of exercise participated in the most', [
        "Walking or jogging", "Running", "Cycling", "Swimming", 
        "Gym", "Yoga or Pilates", "Team sports", "Lifting weights", 
        "Group exercise classes", "Other", "I don't really exercise"
    ])
    diet_barrier = st.selectbox('Barrier to following a healthy balanced diet', [
        "Cost", "Time", "Lack of knowledge on healthy eating", "Temptation and cravings",
        "Ease of access to fast food", "I eat a healthy balanced diet regularly"
    ])
    recommended_friends = st.selectbox('Have you recommended your friends to follow a fitness routine?', ['Yes', 'No'])
    purchased_equipment = st.selectbox('Have you ever purchased fitness equipment?', ['Yes', 'No'])
    motivation = st.selectbox('What motivates you the most to exercise?', [
        "I want to lose weight", "I want to be fit", "I want to gain muscle",
        "I want to improve my health", "I enjoy exercising", "I'm not really interested in exercising"
    ])
    injury_risk = st.selectbox('Injury risk', ['Never', 'Rarely (1-2 times)', 'Occasionally (3-5 times)', 'Often (6+ times)'])
    exercise_most_time = st.selectbox('How do you exercise most of the time?', [
        "Alone", "With friends", "With a trainer", "In a group"
    ])
    exercise_time = st.selectbox('What time of the day do you prefer to exercise?', [
        "Morning", "Afternoon", "Evening", "Night"
    ])
    exercise_duration = st.selectbox('How much time do you spend exercising per day?', [
        "Less than 30 minutes", "30 to 60 minutes", "1 to 2 hours", "More than 2 hours"
    ])
    diet_health = st.selectbox('Would you say, you are following a healthy balanced diet?', ['Yes', 'No'])

    user_data = {
        'Your gender': gender,
        'Your age': age,
        'How do you describe your current level of fitness ?': fitness_level,
        'How often do you exercise?': exercise_frequency,
        'What barrier, if any, prevents you the most from exercising more regularly?': barrier,
        'What form of exercises do you currently participate in the most?': exercise_type,
        'What prevents you the most from following a healthy balanced diet, if any?': diet_barrier,
        'Have you recommended your friends to follow a fitness routine?': recommended_friends,
        'Have you ever purchased fitness equipment?': purchased_equipment,
        'What motivates you the most to exercise?': motivation,
        'Injury risk per year': injury_risk,
        'How do you exercise most of the time?': exercise_most_time,
        'What time of the day do you prefer to exercise?': exercise_time,
        'How much time do you spend exercising per day?': exercise_duration,
        'Would you say, you are following a healthy balanced diet?': diet_health,
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Main function to run the Streamlit app
def main():
    st.title("Injury Type Prediction Dashboard")

    # Add background image
    add_bg_from_local('Baki.jpg')

    # Get user input
    user_input = get_user_input()

    # Button for making prediction
    if st.button('Predict'):
        # One-hot encode the user input
        user_input_encoded = pd.get_dummies(user_input)
        user_input_encoded = user_input_encoded.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(user_input_encoded)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        prevention_link = injury_prevention_links.get(prediction_label, "No link available")
        
        # Display the prediction and the model accuracy with a black background
        st.markdown(f"""
            <div style="background-color: #0E1117; padding: 10px; border-radius: 15px;">
                <p style="color: #FAFAFA;">The predicted injury type is: <b>{prediction_label}</b></p>
                <p style="color: #FAFAFA;">For more information on preventing this type of injury, visit: 
                <a href="{prevention_link}" target="_blank" style="color: #FF4B4B;">Learn how to prevent {prediction_label}</a></p>
                <p style="color: #FAFAFA;">The model prediction accuracy is: <b>{model_accuracy * 100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("Made by: **Aymen Baibai**")

if __name__ == '__main__':
    main()
