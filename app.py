import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64
import pickle
import os
import tempfile
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import our custom audio processor
from utils.audio_utils import AudioProcessor

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .emotion-happy { color: #28a745; font-weight: bold; }
    .emotion-sad { color: #6c757d; font-weight: bold; }
    .emotion-angry { color: #dc3545; font-weight: bold; }
    .emotion-neutral { color: #17a2b8; font-weight: bold; }
    .emotion-fearful { color: #6f42c1; font-weight: bold; }
    .emotion-disgust { color: #fd7e14; font-weight: bold; }
    .emotion-surprised { color: #ffc107; font-weight: bold; }
    .emotion-calm { color: #20c997; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class EmotionRecognitionApp:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.model = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        model_path = "model/emotion_model.h5"
        encoder_path = "model/emotion_model_encoder.pkl"
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.warning("⚠️ Model files not found. Please train the model first using `train_model.py`")
    
    def predict_emotion(self, audio_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            audio = self.audio_processor.load_audio(tmp_path)
            features = self.audio_processor.get_combined_features(audio)
            
            max_frames = 130
            if features.shape[1] < max_frames:
                features = np.pad(features, ((0, 0), (0, max_frames - features.shape[1])), 'constant')
            else:
                features = features[:, :max_frames]
            
            features_reshaped = features.reshape(1, features.shape[0], features.shape[1])
            
            if self.model is not None:
                prediction = self.model.predict(features_reshaped)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                predicted_emotion = self.label_encoder.inverse_transform([predicted_class])[0]
                
                os.unlink(tmp_path)
                
                return predicted_emotion, confidence, features, audio
            else:
                return None, None, features, audio
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None, None, None, None
    
    def plot_audio_waveform(self, audio, sample_rate):
        time = np.linspace(0, len(audio) / sample_rate, len(audio))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name='Audio Waveform',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300,
            showlegend=False
        )
        
        return fig
    
    def plot_mfcc_features(self, features):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('MFCC Features', 'Delta Features', 'Delta-Delta Features'),
            vertical_spacing=0.1
        )
        
        mfcc_features = features[:13, :]
        fig.add_trace(
            go.Heatmap(
                z=mfcc_features,
                colorscale='Viridis',
                name='MFCC',
                showscale=False
            ),
            row=1, col=1
        )
        
        delta_features = features[13:26, :]
        fig.add_trace(
            go.Heatmap(
                z=delta_features,
                colorscale='Plasma',
                name='Delta',
                showscale=False
            ),
            row=2, col=1
        )
        
        delta2_features = features[26:39, :]
        fig.add_trace(
            go.Heatmap(
                z=delta2_features,
                colorscale='Inferno',
                name='Delta-Delta',
                showscale=True
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title="MFCC Feature Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def get_emotion_color(self, emotion):
        color_map = {
            'happy': 'emotion-happy',
            'sad': 'emotion-sad',
            'angry': 'emotion-angry',
            'neutral': 'emotion-neutral',
            'fearful': 'emotion-fearful',
            'disgust': 'emotion-disgust',
            'surprised': 'emotion-surprised',
            'calm': 'emotion-calm'
        }
        return color_map.get(emotion, '')
    
    def run(self):
        st.markdown('<h1 class="main-header">🎤 Emotion Recognition from Speech</h1>', unsafe_allow_html=True)
        
        st.sidebar.markdown("## 📊 About")
        st.sidebar.markdown("""
        This application uses deep learning to recognize emotions from speech audio.
        
        **Supported Emotions:**
        - 😊 Happy
        - 😢 Sad  
        - 😠 Angry
        - 😐 Neutral
        - 😨 Fearful
        - 🤢 Disgust
        - 😲 Surprised
        - 😌 Calm
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">🎵 Upload Audio File</h2>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a .wav audio file",
                type=['wav'],
                help="Upload a speech audio file in WAV format"
            )
            
            if uploaded_file is not None:
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB",
                    "File type": uploaded_file.type
                }
                st.write("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"- {key}: {value}")
                
                predicted_emotion, confidence, features, audio = self.predict_emotion(uploaded_file)
                
                if predicted_emotion is not None:
                    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    emotion_color = self.get_emotion_color(predicted_emotion)
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Emotion: <span class="{emotion_color}">{predicted_emotion.upper()}</span></h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if audio is not None:
                        st.markdown('<h3>📈 Audio Analysis</h3>', unsafe_allow_html=True)
                        
                        st.audio(uploaded_file, format='audio/wav')
                        
                        waveform_fig = self.plot_audio_waveform(audio, self.audio_processor.sample_rate)
                        st.plotly_chart(waveform_fig, use_container_width=True)
                        
                        if features is not None:
                            mfcc_fig = self.plot_mfcc_features(features)
                            st.plotly_chart(mfcc_fig, use_container_width=True)
                
                else:
                    st.error("❌ Could not process the audio file. Please check the file format and try again.")
        
        with col2:
            st.markdown('<h2 class="sub-header">📋 Instructions</h2>', unsafe_allow_html=True)
            st.markdown("""
            1. **Prepare Audio**: Use clear speech audio in WAV format
            2. **Upload File**: Click 'Browse files' and select your audio
            3. **Wait for Processing**: The app will extract features and predict
            4. **View Results**: See the predicted emotion and confidence score
            """)
            
            st.markdown('<h3>🤖 Model Status</h3>', unsafe_allow_html=True)
            if self.model is not None:
                st.success("✅ Model Loaded")
                st.info(f"Classes: {', '.join(self.label_encoder.classes_)}")
            else:
                st.error("❌ Model Not Found")
                st.info("Please run `python train_model.py` to train the model first.")

def main():
    app = EmotionRecognitionApp()
    app.run()

if __name__ == "__main__":
    main() 