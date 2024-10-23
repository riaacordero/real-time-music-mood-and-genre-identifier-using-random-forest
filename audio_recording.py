import sounddevice as sd
import numpy as np
from model import load_trained_model, extract_features

def record_audio(duration=8, sr=22050):
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio

def main():
    # Load the trained model
    model = load_trained_model()
    if model is None:
        return

    # Record audio
    audio_data = record_audio(duration=8)  # Duration in seconds
    features = extract_features(audio_data)

    # Make prediction
    if features is not None:
        genre_prediction = model.predict([features])
        print(f"Predicted Genre: {genre_prediction[0]}")
    else:
        print("Error in feature extraction.")

if __name__ == "__main__":
    main()
