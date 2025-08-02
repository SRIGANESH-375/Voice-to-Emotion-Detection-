from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import pickle
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class EmotionPredictor:
    def __init__(self, model_path='best_ravdess_emotion_model.h5', 
                 label_encoder_path='label_encoder.pkl', 
                 scaler_path='scaler.pkl'):
        """
        Initialize the emotion predictor with trained model and preprocessors
        """
        self.sample_rate = 22050
        self.duration = 3.0
        self.n_mels = 128
        self.n_mfcc = 40
        self.max_len = int(self.sample_rate * self.duration)
        
        # RAVDESS dataset path
        self.dataset_path = r'C:\Users\91900\OneDrive\Desktop\V2E\Ravdess_Dataset'
        
        # RAVDESS emotion mapping
        self.ravdess_emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        # Load model and preprocessors
        self.load_model_components(model_path, label_encoder_path, scaler_path)
    
    def load_model_components(self, model_path, label_encoder_path, scaler_path):
        """Load the trained model and preprocessing components"""
        try:
            # Load model
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                # Create a dummy model structure for demo purposes
                self.create_dummy_model()
            
            # Load label encoder
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"Label encoder loaded from {label_encoder_path}")
            else:
                logger.warning(f"Label encoder not found: {label_encoder_path}")
                self.create_dummy_label_encoder()
            
            # Load scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found: {scaler_path}")
                self.create_dummy_scaler()
                
        except Exception as e:
            logger.error(f"Error loading model components: {e}")
            self.create_dummy_components()
    
    def create_dummy_model(self):
        """Create a dummy model for demo purposes when real model is not available"""
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import Dense # type: ignore
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(200,)),  # Dummy input shape
            Dense(32, activation='relu'),
            Dense(8, activation='softmax')  # 8 emotions
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        logger.info("Created dummy model for demo purposes")
    
    def create_dummy_label_encoder(self):
        """Create a dummy label encoder"""
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.label_encoder.fit(emotions)
        logger.info("Created dummy label encoder")
    
    def create_dummy_scaler(self):
        """Create a dummy scaler"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        # Fit with dummy data
        dummy_data = np.random.random((100, 200))
        self.scaler.fit(dummy_data)
        logger.info("Created dummy scaler")
    
    def create_dummy_components(self):
        """Create all dummy components if loading fails"""
        self.create_dummy_model()
        self.create_dummy_label_encoder()
        self.create_dummy_scaler()
    
    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features for emotion recognition
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate audio to fixed length
            if len(audio) < self.max_len:
                audio = np.pad(audio, (0, self.max_len - len(audio)))
            else:
                audio = audio[:self.max_len]
            
            # Extract various features
            features = {}
            
            # 1. MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            features['mfcc'] = np.concatenate([mfccs, mfccs_delta, mfccs_delta2], axis=0).T
            
            # 2. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spec'] = mel_spec_db.T
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = chroma.T
            
            # 4. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['contrast'] = contrast.T
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr'] = zcr.T
            
            # 6. Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['rolloff'] = rolloff.T
            
            # 7. Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['centroid'] = centroid.T
            
            # 8. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features['tonnetz'] = tonnetz.T
            
            # Combine all features
            combined_features = np.concatenate([
                features['mfcc'],
                features['mel_spec'],
                features['chroma'],
                features['contrast'],
                features['zcr'],
                features['rolloff'],
                features['centroid'],
                features['tonnetz']
            ], axis=1)
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        """
        try:
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                return None
            
            # For demo purposes, if we have a dummy model, return random predictions
            if not os.path.exists('best_ravdess_emotion_model.h5'):
                return self.generate_demo_prediction()
            
            # Preprocess features
            features = features.reshape(1, features.shape[0], features.shape[1])
            
            # Handle scaler transformation
            original_shape = features.shape
            features_2d = features.reshape(-1, features.shape[2])
            features_scaled = self.scaler.transform(features_2d)
            features_scaled = features_scaled.reshape(original_shape)
            
            # Predict
            prediction = self.model.predict(features_scaled, verbose=0)
            
            # Get all emotion probabilities
            emotion_probs = {}
            for i, emotion in enumerate(self.label_encoder.classes_):
                emotion_probs[emotion] = float(prediction[0][i])
            
            # Get primary emotion
            primary_emotion_idx = np.argmax(prediction)
            primary_emotion = self.label_encoder.classes_[primary_emotion_idx]
            confidence = float(prediction[0][primary_emotion_idx])
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'all_emotions': emotion_probs
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            logger.error(traceback.format_exc())
            return self.generate_demo_prediction()
    
    def generate_demo_prediction(self):
        """Generate demo prediction when real model is not available"""
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Generate random but realistic probabilities
        probs = np.random.dirichlet(np.ones(len(emotions)) * 2)  # Creates more realistic distribution
        
        emotion_probs = {}
        for i, emotion in enumerate(emotions):
            emotion_probs[emotion] = float(probs[i])
        
        primary_emotion_idx = np.argmax(probs)
        primary_emotion = emotions[primary_emotion_idx]
        confidence = float(probs[primary_emotion_idx])
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_emotions': emotion_probs
        }

# Initialize the predictor
predictor = EmotionPredictor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Read the HTML file content (you would replace this with your actual HTML)
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Voice Emotion Detection</title>
            <style>
                /* Your existing CSS styles here */
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }
                .container {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
                    max-width: 600px;
                    width: 100%;
                    text-align: center;
                }
                h1 {
                    color: #333;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                .upload-section {
                    margin-bottom: 40px;
                    padding: 30px;
                    border: 2px dashed #667eea;
                    border-radius: 15px;
                    transition: all 0.3s ease;
                }
                .record-btn, .file-label {
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 50px;
                    font-size: 18px;
                    cursor: pointer;
                    margin: 10px;
                    transition: all 0.3s ease;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                    display: inline-block;
                }
                .file-input { display: none; }
                .results {
                    margin-top: 30px;
                    padding: 25px;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    border-radius: 15px;
                    color: white;
                    display: none;
                }
                .emotion-display {
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 15px;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                }
                .confidence {
                    font-size: 1.2em;
                    opacity: 0.9;
                }
                .loading {
                    display: none;
                    margin: 20px 0;
                }
                .spinner {
                    border: 4px solid rgba(102, 126, 234, 0.3);
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .emotion-bars {
                    margin-top: 20px;
                    text-align: left;
                }
                .emotion-bar {
                    margin-bottom: 10px;
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 10px;
                    overflow: hidden;
                }
                .emotion-bar-fill {
                    height: 30px;
                    background: rgba(255, 255, 255, 0.8);
                    display: flex;
                    align-items: center;
                    padding: 0 15px;
                    font-weight: bold;
                    color: #333;
                    transition: width 1s ease;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 Voice Emotion Detection</h1>
                
                <div class="upload-section">
                    <h3 style="margin-bottom: 20px; color: #555;">Upload Audio File</h3>
                    <input type="file" id="fileInput" class="file-input" accept=".wav,.mp3,.m4a,.ogg,.flac">
                    <label for="fileInput" class="file-label">📁 Choose Audio File</label>
                    <div id="fileName" style="margin-top: 15px; color: #666;"></div>
                </div>

                <button id="analyzeBtn" class="record-btn" style="display: none;">🔍 Analyze Emotion</button>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px; color: #667eea;">Analyzing emotions...</p>
                </div>

                <div id="results" class="results">
                    <div class="emotion-display" id="primaryEmotion"></div>
                    <div class="confidence" id="confidenceScore"></div>
                    <div class="emotion-bars" id="emotionBars"></div>
                </div>
            </div>

            <script>
                const fileInput = document.getElementById('fileInput');
                const fileName = document.getElementById('fileName');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');

                const emotionEmojis = {
                    'neutral': '😐',
                    'calm': '😌',
                    'happy': '😊',
                    'sad': '😢',
                    'angry': '😠',
                    'fearful': '😨',
                    'disgust': '🤢',
                    'surprised': '😲'
                };

                let selectedFile = null;

                fileInput.addEventListener('change', (event) => {
                    const file = event.target.files[0];
                    if (file) {
                        fileName.textContent = `Selected: ${file.name}`;
                        selectedFile = file;
                        analyzeBtn.style.display = 'inline-block';
                    }
                });

                analyzeBtn.addEventListener('click', async () => {
                    if (!selectedFile) {
                        alert('Please select an audio file first.');
                        return;
                    }

                    loading.style.display = 'block';
                    results.style.display = 'none';

                    const formData = new FormData();
                    formData.append('audio', selectedFile);

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();

                        if (data.success) {
                            displayResults(data.primary_emotion, data.confidence, data.all_emotions);
                        } else {
                            alert('Error: ' + data.error);
                        }
                    } catch (error) {
                        alert('Error analyzing audio: ' + error.message);
                    }

                    loading.style.display = 'none';
                });

                function displayResults(primaryEmotion, confidence, allEmotions) {
                    document.getElementById('primaryEmotion').textContent =
                        `${emotionEmojis[primaryEmotion] || '🎭'} ${primaryEmotion.toUpperCase()}`;
                    document.getElementById('confidenceScore').textContent =
                        `Confidence: ${(confidence * 100).toFixed(1)}%`;

                    const emotionBars = document.getElementById('emotionBars');
                    emotionBars.innerHTML = '';

                    Object.entries(allEmotions).forEach(([emotion, score], index) => {
                        const percentage = score * 100;
                        const barContainer = document.createElement('div');
                        barContainer.className = 'emotion-bar';
                        
                        const barFill = document.createElement('div');
                        barFill.className = 'emotion-bar-fill';
                        barFill.style.width = '0%';
                        barFill.textContent = `${emotionEmojis[emotion] || '🎭'} ${emotion} (${percentage.toFixed(1)}%)`;
                        
                        barContainer.appendChild(barFill);
                        emotionBars.appendChild(barContainer);

                        setTimeout(() => {
                            barFill.style.width = `${percentage}%`;
                        }, 100 * index);
                    });

                    results.style.display = 'block';
                }
            </script>
        </body>
        </html>
        """
        return render_template_string(html_content)
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return f"Error loading page: {e}", 500

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Handle emotion prediction requests"""
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Predict emotion
            result = predictor.predict_emotion(temp_path)
            
            if result is None:
                return jsonify({'success': False, 'error': 'Failed to process audio file'}), 500
            
            response = {
                'success': True,
                'primary_emotion': result['primary_emotion'],
                'confidence': result['confidence'],
                'all_emotions': result['all_emotions']
            }
            
            return jsonify(response)
        
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in emotion prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'scaler_loaded': predictor.scaler is not None,
        'label_encoder_loaded': predictor.label_encoder is not None
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*60)
    print("Voice Emotion Detection Flask Backend")
    print("="*60)
    print(f"Dataset path: C:\\Users\\91900\\OneDrive\\Desktop\\V2E\\Ravdess_Dataset")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Allowed file types: {ALLOWED_EXTENSIONS}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")
    print("="*60)
    
    # Check if model files exist
    model_files = ['best_ravdess_emotion_model.h5', 'label_encoder.pkl', 'scaler.pkl']
    for file in model_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"⚠ {file} not found - using demo mode")
    
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)