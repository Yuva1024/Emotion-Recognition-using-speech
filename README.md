# 🎤 Emotion Recognition from Speech

A deep learning project that recognizes emotions from speech audio using MFCC features and a CNN model. Built with Python, TensorFlow/Keras, and Streamlit.

## 🎯 Features

- **8 Emotion Classes**: Happy, Sad, Angry, Neutral, Fearful, Disgust, Surprised, Calm
- **MFCC Feature Extraction**: Advanced audio processing using librosa
- **CNN Deep Learning Model**: Convolutional Neural Network for emotion classification
- **Streamlit Web App**: User-friendly interface for audio upload and prediction
- **Real-time Processing**: Instant emotion prediction with confidence scores
- **Audio Visualization**: Waveform and MFCC feature plots

## 📁 Project Structure

```
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── utils/
│   └── audio_utils.py     # Audio processing utilities
├── model/
│   ├── emotion_model.h5   # Trained model (generated after training)
│   └── emotion_model_encoder.pkl  # Label encoder (generated after training)
├── ravdess_speech/        # Dataset folder (add your audio files here)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv emotion_env

# Activate virtual environment
# On Windows:
emotion_env\Scripts\activate
# On macOS/Linux:
source emotion_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the RAVDESS dataset and extract it to the `ravdess_speech` folder:

```bash
# Create dataset directory
mkdir ravdess_speech

# Download RAVDESS dataset from:
# https://zenodo.org/record/1188976
# Extract the audio files to ravdess_speech/
```

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the RAVDESS dataset
- Extract MFCC features from audio files
- Train a CNN model for emotion recognition
- Save the trained model to `model/emotion_model.h5`
- Generate training plots and evaluation metrics

### 5. Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎵 How to Use the App

1. **Upload Audio**: Click "Browse files" and select a `.wav` audio file
2. **Wait for Processing**: The app extracts MFCC features automatically
3. **View Results**: See the predicted emotion and confidence score
4. **Analyze Audio**: View waveform and MFCC feature visualizations

## 🔧 Technical Details

### Audio Processing
- **Sample Rate**: 22050 Hz
- **Duration**: 3 seconds (padded/truncated)
- **Features**: MFCC + Delta + Delta-Delta (39 coefficients total)

### Model Architecture
- **Input**: (39, 130) feature matrix
- **Architecture**: 3 Convolutional blocks + Dense layers
- **Output**: 8 emotion classes with softmax activation
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout and Batch Normalization

### Supported Audio Formats
- **Primary**: WAV files
- **Sample Rate**: Any (automatically resampled to 22050 Hz)
- **Duration**: Variable (automatically processed to 3 seconds)

## 📊 Model Performance

The model typically achieves:
- **Training Accuracy**: 85-90%
- **Validation Accuracy**: 80-85%
- **Test Accuracy**: 75-80%

Performance may vary based on:
- Dataset quality and size
- Audio recording conditions
- Speaker characteristics

## 🛠️ Customization

### Modify Model Architecture
Edit `train_model.py` to change:
- Number of convolutional layers
- Filter sizes and counts
- Dense layer dimensions
- Dropout rates

### Adjust Audio Processing
Edit `utils/audio_utils.py` to modify:
- Sample rate
- Audio duration
- MFCC parameters
- Feature extraction methods

### Add New Emotions
1. Update emotion mapping in `AudioProcessor`
2. Retrain the model with new data
3. Update the Streamlit app styling

## 📝 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM
- GPU recommended for training (optional)

### Python Packages
- TensorFlow 2.15.0
- Keras 2.15.0
- librosa 0.10.1
- Streamlit 1.28.1
- NumPy 1.24.3
- Pandas 2.0.3
- scikit-learn 1.3.0
- Plotly 5.17.0

## 🔍 Troubleshooting

### Common Issues

**1. Model not found error**
```bash
# Ensure you've trained the model first
python train_model.py
```

**2. Audio processing errors**
- Check audio file format (WAV recommended)
- Ensure audio contains speech (not music/noise)
- Verify file is not corrupted

**3. Memory issues during training**
- Reduce batch size in `train_model.py`
- Use smaller audio duration
- Process dataset in smaller chunks

**4. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Tips

1. **For Training**:
   - Use GPU if available
   - Increase batch size for faster training
   - Use data augmentation for better generalization

2. **For Inference**:
   - Use shorter audio clips for faster processing
   - Optimize model for deployment if needed

## 📚 Dataset Information

### RAVDESS Dataset
- **Source**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **Format**: 24-bit, 48kHz WAV files
- **Emotions**: 8 emotional states
- **Speakers**: 24 professional actors
- **License**: Creative Commons Attribution-NonCommercial 4.0

### Emotion Mapping
- 01: Neutral
- 02: Calm
- 03: Happy
- 04: Sad
- 05: Angry
- 06: Fearful
- 07: Disgust
- 08: Surprised

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- RAVDESS dataset creators
- TensorFlow/Keras team
- Streamlit developers
- librosa contributors

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Emotion Recognition! 🎤😊** 