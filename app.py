import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("results/model.h5")

# Define the label dictionary
label2int = {
    "male": 0,
    "female": 1
}
int2label = {v: k for k, v in label2int.items()}

def extract_features(audio_path, vector_length=128):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=vector_length)
    mfcc = np.mean(mfcc.T, axis=0)  # Averaging across time axis to get a fixed-size vector
    return mfcc

# Streamlit app
st.title("Gender Recognition from Voice")
st.write("Upload an audio sample to predict the gender.")

# File uploader
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if audio_file is not None:
    # Save the uploaded file temporarily
    audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

    # Extract features
    try:
        features = extract_features(audio_path)
        features = features.reshape(1, -1)  # Reshape for model input

        # Make prediction
        prediction = model.predict(features)
        gender = int2label[int(round(prediction[0][0]))]

        # Show result
        st.write(f"Predicted Gender: **{gender.capitalize()}**")

        # Play the uploaded audio
        st.audio(audio_path, format="audio/wav")
    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
    finally:
        # Clean up temp file
        os.remove(audio_path)