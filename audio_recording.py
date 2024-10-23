import sounddevice as sd
import numpy as np

def record_audio(duration=10, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

# Test function
if __name__ == "__main__":
    audio_data = record_audio(duration=5)  # Duration in seconds
    print(audio_data) 
