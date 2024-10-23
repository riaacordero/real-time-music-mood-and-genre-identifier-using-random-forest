import sounddevice as sd
import numpy as np
import librosa

def record_audio(duration=10, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def extract_features(audio, sr=22050):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)  # Mean across time

# Test function
if __name__ == "__main__":
    audio_data = record_audio(duration=5)  # Duration in seconds
    features = extract_features(audio_data)
    print(audio_data) 
