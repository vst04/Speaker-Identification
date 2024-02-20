import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Function to extract features from an audio file
def extract_features(file_path, num_mfcc=13, n_fft=2048, hop_length=512, max_frames=None):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Pad or truncate to have the same number of frames
    if max_frames is not None:
        if len(mfcc[0]) < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - len(mfcc[0]))), mode='constant')
        else:
            mfcc = mfcc[:, :max_frames]

    return mfcc.T  # Transpose MFCC matrix

# Load the trained model
model = load_model('C:\\speakerrec2\\speaker_recognition_model.h5')  # Replace with the actual path to your trained model

# Load the label encoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('C:\\speakerrec2\\label_encoder_classes.npy')  # Replace with the actual path

# Example of using the trained model for prediction
new_audio_file_path = 'C:\\speakerrec2\\16000_pcm_speeches\\Jens_Stoltenberg\\2.wav'

# Extract features from the new audio file
max_frames = model.input_shape[1]
new_audio_features = extract_features(new_audio_file_path, max_frames=max_frames)
new_audio_features = np.expand_dims(new_audio_features, axis=0)
new_audio_features = new_audio_features[..., np.newaxis]

# Make predictions using the trained model
prediction = model.predict(new_audio_features)
predicted_label_index = np.argmax(prediction)
predicted_label = label_encoder.classes_[predicted_label_index]

print(f'Predicted Speaker: {predicted_label}')