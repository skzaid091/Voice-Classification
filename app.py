import streamlit as st
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

import time


###########  WITHOUT PICKLE  ###############################
# Load the model
model = tf.keras.models.load_model('Models/audio_classification.keras')

# Load the label encoder
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('Models/label_encoder.npy')


# ###########  WITH PICKLE  ###############################
# with open('Models/audio_classification.pkl', 'rb') as f:
#     audio_classification = pickle.load(f)
# model = tf.keras.models.load_model(audio_classification)

# with open('Models/label_encoder.pkl', 'rb') as f:
#     classes = pickle.load(f)
# labelencoder = LabelEncoder()
# labelencoder.classes_ = np.load('Models/label_encoder.npy')


# Function to extract features from audio file
def features_extractor(audio_file):
    audio, sample_rate = librosa.load(audio_file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def transform(text):
    temp = text.replace("_", " ")
    return temp.title()
            

# Streamlit UI
st.title("Audio Classification")
st.write("(Updated Version are COMING SOON....)")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Extract features from the uploaded audio file
    features = features_extractor("temp_audio.wav")
    result = features.reshape(1, -1)

    # Make prediction
    prediction = model.predict(result)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode predicted class
    answer = labelencoder.inverse_transform(predicted_class)
    answer = transform(answer[0])

    # # Display uploaded audio file
    # st.write("                                  ")
    # st.audio("temp_audio.wav", format="audio/wav")

    progress_bar = st.progress(0)
    for i in range(100):  # Simulate prediction progress (replace with actual logic)
        time.sleep(0.01)  # Artificial delay
        progress_bar.progress(i + 1)
    
    # Display prediction result
    st.write("                                  ")
    st.subheader("Predicted Class : " + answer)
