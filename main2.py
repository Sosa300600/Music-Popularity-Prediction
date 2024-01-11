import pandas as pd
import numpy as np 
import streamlit as st 
import joblib


# Load model & preprocessor
artifact = joblib.load('song_popularity_pred.joblib')

st.title('Music Popularity Prediction App')


song_duration_ms = st.number_input("Song duration in milliseconds", format='%.1f')

acousticness = st.number_input("Acousticness", format='%.1f')

danceability = st.number_input("Danceability", format='%.1f')

energy = st.number_input("Energy", format='%.1f')

instrumentalness = st.number_input("Instrumentalness", format='%.1f')

key = st.selectbox("Key", ["", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

liveness = st.number_input("Liveness", format='%.1f')

loudness = st.number_input("Loudness", format='%.1f')

audio_mode = st.selectbox("Audio mode", ['', "Major", "Minor"])

speechiness = st.number_input("Speechiness", format='%.1f')

tempo = st.number_input("Tempo", format='%.1f')

time_signature = st.selectbox("Time-Signature", ["", 2, 3, 4, 5])

audio_valence = st.number_input("Audio-Valence", format='%.1f')


user_input = {
'song_duration_ms': song_duration_ms,
'acousticness': acousticness,
'danceability': danceability,
'energy': energy,
'instrumentalness': instrumentalness,
'key': key,
'liveness': liveness,
'loudness': loudness,
'audio_mode': 1 if audio_mode=='Major' else 0,
'speechiness': speechiness,
'tempo': tempo,
'time_signature': time_signature,
'audio_valence': audio_valence}


def check_missing_inputs(data):
    return any(value == '' for value in data.values())


def make_prediction(data):
    df = pd.DataFrame([data])
    X = pd.DataFrame(artifact['preprocessing'].transform(df),
                     columns=artifact['preprocessing'].get_feature_names_out())
    
    prediction = artifact['model'].predict(X)
    pred = 'Popular' if prediction == 1 else 'Not Popular'
    return pred

# Button to make prediction
if st.button('Predict Music Popularity'):
    if check_missing_inputs(user_input):
        st.error("Please fill in all inputs.")
    else:
        result = make_prediction(user_input)
        st.success(result)