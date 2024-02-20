import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

# Function to load audio files and extract features
def load_data_from_folder(folder_path, num_mfcc=13, sr=44100, n_fft=2048, hop_length=512):
    mfccs = []
    labels = []
    for speaker_folder in os.listdir(folder_path):
        speaker_path = os.path.join(folder_path, speaker_folder)
        if os.path.isdir(speaker_path):
            for filename in os.listdir(speaker_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(speaker_path, filename)
                    audio, sr = librosa.load(file_path, sr=sr)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfccs.append(mfcc.T)  # Transpose MFCC matrix
                    labels.append(speaker_folder)
    return mfccs, labels

# Path to the folder containing audio files
folder_path = 'C:\\speakerrec2\\16000_pcm_speeches'

# Load data and preprocess
mfccs, labels = load_data_from_folder(folder_path)

# Convert labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Pad or truncate each MFCC to have the same number of frames
max_frames = max(len(mfcc) for mfcc in mfccs)
mfccs_padded = [np.pad(mfcc, ((0, max_frames - len(mfcc)), (0, 0)), mode='constant') for mfcc in mfccs]

# Convert the list of padded MFCCs to a NumPy array
mfccs_array = np.array(mfccs_padded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mfccs_array, encoded_labels, test_size=0.2, random_state=42)

# Reshape MFCC arrays for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(labels)), activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)