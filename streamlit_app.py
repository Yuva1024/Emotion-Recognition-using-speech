import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import librosa
import os
import tempfile
import pickle
import warnings
import tensorflow as tf

# Import our custom audio processor
from utils.audio_utils import AudioProcessor

# Custom InputLayer to handle compatibility issues
class CustomInputLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape=None, batch_shape=None, dtype=None, sparse=False, ragged=False, name=None, **kwargs):
        # Remove batch_shape from kwargs to avoid the error
        if 'batch_shape' in kwargs:
            del kwargs['batch_shape']
        super().__init__(name=name, **kwargs)
        self._input_shape = input_shape
        self._batch_shape = batch_shape
        self._dtype = dtype
        self._sparse = sparse
        self._ragged = ragged
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def batch_shape(self):
        return self._batch_shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def sparse(self):
        return self._sparse
    
    @property
    def ragged(self):
        return self._ragged

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
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #34495e;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .emotion-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .confidence-container {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
    .confidence-fill {
        height: 8px;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .info-card {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .upload-area {
        background: rgba(102, 126, 234, 0.1);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px dashed #667eea;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Load the trained emotion recognition model and label encoder.
    """
    try:
        # Use the best simple model by default (more compatible)
        model_path = 'model/best_simple_model.h5'
        encoder_path = 'model/emotion_model_simple_encoder.pkl'
        
        if not os.path.exists(model_path):
            # Fallback to GPU model
            model_path = 'model/best_gpu_model.h5'
            encoder_path = 'model/emotion_model_encoder.pkl'
            
        if not os.path.exists(model_path):
            # Fallback to original model
            model_path = 'model/emotion_model.h5'
            encoder_path = 'model/emotion_model_encoder.pkl'
            
        if not os.path.exists(model_path):
            st.error("❌ No model file found! Please train the model first.")
            return None, None, None
            
        # Load model with compatibility handling
        try:
            # First try loading without custom objects (works on local machine)
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            # If that fails, try with custom objects to handle deprecated parameters
            try:
                custom_objects = {
                    'InputLayer': CustomInputLayer
                }
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e2:
                # If that fails, try with skip_serialization_validation
                try:
                    model = tf.keras.models.load_model(model_path, compile=False, skip_serialization_validation=True)
                except Exception as e3:
                    # Last resort: try with experimental options
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
                    except Exception as e4:
                        st.error(f"❌ Failed to load model: {str(e4)}")
                        return None, None, None
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Determine model type
        model_input_shape = model.input_shape
        if len(model_input_shape) == 3:  # CNN model
            model_type = "CNN"
        else:  # Simple model
            model_type = "Simple"
            
        st.success(f"✅ {model_type} model loaded successfully!")
        st.info(f"📊 Model: {os.path.basename(model_path)}")
        
        return model, label_encoder, model_type
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def process_audio(uploaded_file):
    """Process uploaded audio file and extract features."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract features
        processor = AudioProcessor()
        audio = processor.load_audio(tmp_path)
        features = processor.get_combined_features(audio)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return features, uploaded_file
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        return None, None

def predict_emotion(model, label_encoder, features, model_type):
    """
    Predict emotion from extracted features.
    """
    try:
        # Normalize features
        features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Reshape for model input
        if model_type == "Simple":
            features_reshaped = features_normalized.reshape(1, -1)
        else:
            features_reshaped = features_normalized.reshape(1, features_normalized.shape[0], features_normalized.shape[1])
        
        # Make prediction
        predictions = model.predict(features_reshaped, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[0][predicted_class_idx]
        
        # Get all class probabilities
        emotion_probs = {}
        for i, emotion in enumerate(label_encoder.classes_):
            emotion_probs[emotion] = float(predictions[0][i])
        
        return predicted_emotion, confidence, emotion_probs
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None, None

def display_results(predicted_emotion, confidence, all_probabilities, label_encoder):
    """Display prediction results with professional styling."""
    
    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
    
    # Prediction box with emoji
    emoji = get_emotion_emoji(predicted_emotion)
    st.markdown(f"""
    <div class="prediction-box">
        <div class="emotion-emoji">{emoji}</div>
        <h3 style="font-size: 2rem; margin-bottom: 1rem;">{predicted_emotion.title()}</h3>
        <p style="font-size: 1.2rem; opacity: 0.9;">Confidence: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence scores for all emotions
    st.markdown('<h3 class="sub-header">📊 Confidence Breakdown</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="confidence-container">
    """, unsafe_allow_html=True)
    
    # Sort emotions by confidence
    sorted_emotions = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for emotion, prob in sorted_emotions:
        # Color coding based on confidence
        if prob > 0.7:
            color = "#28a745"  # Green for high confidence
        elif prob > 0.4:
            color = "#ffc107"  # Yellow for medium confidence
        else:
            color = "#dc3545"  # Red for low confidence
            
        # Highlight predicted emotion
        if emotion == predicted_emotion:
            color = "#667eea"  # Blue for predicted emotion
        
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-text" style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span><strong>{emotion.title()}</strong></span>
                <span>{prob:.1%}</span>
            </div>
            <div class="confidence-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {color}, {color}dd);"></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def get_emotion_emoji(emotion):
    """Get emoji for emotion."""
    emoji_map = {
        'angry': '😠',
        'calm': '😌',
        'fearful': '😨',
        'happy': '😊',
        'neutral': '😐',
        'sad': '😢'
    }
    return emoji_map.get(emotion.lower(), '🎤')

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Emotion Recognition from Speech</h1>', unsafe_allow_html=True)
    
    # Load model
    model, label_encoder, model_type = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.markdown('<h3 class="sub-header">ℹ️ Information</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📋 Model Information</h4>
            <p><strong>Architecture:</strong> {model_type}</p>
            <p><strong>Input Shape:</strong> {input_shape}</p>
            <p><strong>Classes:</strong> {num_classes}</p>
        </div>
        """.format(
            model_type=model_type,
            input_shape=model.input_shape,
            num_classes=len(label_encoder.classes_) if label_encoder else "Unknown"
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>💡 Tips for Best Results</h4>
            <ul>
                <li>Use clear, high-quality audio</li>
                <li>Speak naturally and expressively</li>
                <li>Record in a quiet environment</li>
                <li>Keep audio length between 2-5 seconds</li>
                <li>Supported formats: WAV, MP3, M4A</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎵 Upload Audio File</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="upload-area">
            <p><strong>Upload your audio file to analyze the emotional content</strong></p>
            <p>Supported formats: WAV, MP3, M4A, FLAC</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload an audio file to analyze emotions"
        )
        
        if uploaded_file is not None:
            # Display uploaded file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            st.markdown("""
            <div class="info-card">
                <h4>📁 File Information</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Process audio
            try:
                # Process audio
                features, uploaded_file = process_audio(uploaded_file)
                
                if features is not None:
                    # Display audio player
                    st.markdown('<h3 class="sub-header">🎧 Audio Preview</h3>', unsafe_allow_html=True)
                    st.audio(uploaded_file, format='audio/wav')
                    
                    # Make prediction
                    predicted_emotion, confidence, all_probabilities = predict_emotion(
                        model, label_encoder, features, model_type
                    )
                    
                    # Display results
                    display_results(predicted_emotion, confidence, all_probabilities, label_encoder)
                    
                else:
                    st.error("❌ Failed to extract features from audio file.")
                    
            except Exception as e:
                st.error(f"❌ Error processing audio file: {str(e)}")
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Quick Stats</h2>', unsafe_allow_html=True)
        
        # Emotion distribution chart
        if label_encoder is not None:
            emotions = label_encoder.classes_
            performance_data = {
                'Happy': 83.8,
                'Angry': 78.4,
                'Calm': 78.4,
                'Fearful': 59.5,
                'Neutral': 55.6,
                'Sad': 40.5
            }
            
            # Create performance chart
            fig = px.bar(
                x=list(performance_data.keys()),
                y=list(performance_data.values()),
                title="Model Performance by Emotion",
                labels={'x': 'Emotion', 'y': 'Accuracy (%)'},
                color=list(performance_data.values()),
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                title_font_size=16,
                title_font_color='#2c3e50',
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🎤 <strong>Emotion Recognition from Speech</strong></p>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Librosa</p>
        <p>Dataset: RAVDESS Speech Emotion Recognition</p>
        <p>Model: Deep Learning with MFCC Features</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 