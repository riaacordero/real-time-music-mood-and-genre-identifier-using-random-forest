import os, librosa, tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATASET_PATH = './dataset/genres_original'

if os.path.exists(DATASET_PATH):
    print("Dataset path found.")
else:
    print("Dataset path not found.")

def extract_features_from_file(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset():
    features = []
    labels = []
    genres = os.listdir(DATASET_PATH)

    print("Found genres:", genres)  # Debugging step
    
    for genre in tqdm.tqdm(genres, desc="Processing Genres"):
        genre_path = os.path.join(DATASET_PATH, genre)
        if os.path.isdir(genre_path):
            for filename in tqdm.tqdm(os.listdir(genre_path), desc=f"Processing {genre}", leave=False):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path, filename)
                    # print(f"Processing file: {file_path}")  # Debugging step
                    mfccs = extract_features_from_file(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(genre)

    return np.array(features), np.array(labels)

# Load features and labels
features, labels = load_dataset()

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

with tqdm.tqdm(total=100, desc="Training Model") as pbar:
    # Train a Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pbar.update(100)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
