from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model, scaler, and label encoder
model = load_model('voice_emotion_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.util.fix_length(y, size=sr * 3)  # pad or trim to 3 seconds
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # shape should be (173, 40)
        if mfcc.shape != (173, 40):
            print(f"❌ Invalid shape: {mfcc.shape}")
            return None
        return mfcc
    except Exception as e:
        print(f"🔥 Error in extract_features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return "No audio file found", 400

    file = request.files['audio_data']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"📁 File saved at: {file_path}")

    features = extract_features(file_path)
    if features is None:
        return "Feature extraction failed", 400

    # Reshape and scale
    features = features.reshape(1, 173, 40)
    try:
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        return jsonify({'emotion': predicted_label})
    except Exception as e:
        print(f"🔥 Prediction error: {e}")
        return "Prediction failed", 500

if __name__ == '__main__':
    app.run(debug=True)
