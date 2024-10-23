import sounddevice as sd
import numpy as np
import librosa

def record_audio(duration=8, sr=22050):
    try:
        print(f"Recording {duration} seconds of audio...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        audio = audio.flatten()
        
        if len(audio) == 0:
            raise ValueError("No audio data recorded. Please check the microphone.")

        # Normalize audio to have consistent volume
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def extract_features(audio, sr=22050):
    try:
        if audio is None:
            raise ValueError("No audio input for feature extraction.")

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)  # Mean across time

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Test function
if __name__ == "__main__":
    audio_data = record_audio(duration=5)  # Duration in seconds
    if audio_data is not None:
        features = extract_features(audio_data)
        print(f"Extracted features: {features}")
