import librosa
import numpy as np
import os
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing utilities for emotion recognition from speech.
    Handles audio loading, preprocessing, and MFCC feature extraction.
    """
    
    def __init__(self, sample_rate: int = 22050, duration: float = 3.0):
        """
        Initialize AudioProcessor with configuration parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            duration (float): Target duration in seconds for audio clips
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Emotion mapping for RAVDESS dataset
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm', 
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        logger.info(f"AudioProcessor initialized with sample_rate={sample_rate}, duration={duration}")
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Audio signal as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {file_path}, shape: {audio.shape}, sample_rate: {sr}")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio signal by padding or truncating to target length.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Preprocessed audio signal
        """
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            # Pad with zeros if audio is too short
            audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
        else:
            # Truncate if audio is too long
            audio = audio[:self.target_length]
        
        logger.info(f"Preprocessed audio shape: {audio.shape}")
        return audio
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13, n_fft: int = 2048, 
                    hop_length: int = 512) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            n_mfcc (int): Number of MFCC coefficients
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            
        Returns:
            np.ndarray: MFCC features (n_mfcc, time_frames)
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Normalize MFCC features
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            
            logger.info(f"Extracted MFCC features: {mfcc.shape}")
            return mfcc
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            raise
    
    def extract_delta_features(self, mfcc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract delta and delta-delta features from MFCC.
        
        Args:
            mfcc (np.ndarray): MFCC features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Delta and delta-delta features
        """
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        logger.info(f"Extracted delta features: {delta.shape}, delta2: {delta2.shape}")
        return delta, delta2
    
    def get_combined_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract combined MFCC, delta, and delta-delta features.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Combined features (3 * n_mfcc, time_frames)
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        
        # Extract MFCC features
        mfcc = self.extract_mfcc(audio)
        
        # Extract delta features
        delta, delta2 = self.extract_delta_features(mfcc)
        
        # Combine all features
        combined_features = np.vstack([mfcc, delta, delta2])
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def get_emotion_from_filename(self, filename: str) -> str:
        """
        Extract emotion label from RAVDESS filename.
        RAVDESS filename format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        
        Args:
            filename (str): RAVDESS audio filename
            
        Returns:
            str: Emotion label
        """
        try:
            # Extract emotion code from filename
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                return self.emotion_map.get(emotion_code, 'unknown')
            else:
                logger.warning(f"Invalid filename format: {filename}")
                return 'unknown'
        except Exception as e:
            logger.error(f"Error extracting emotion from filename {filename}: {str(e)}")
            return 'unknown'
    
    def load_dataset(self, dataset_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load audio files from dataset directory and extract features.
        
        Args:
            dataset_path (str): Path to dataset directory
            
        Returns:
            Tuple[List[np.ndarray], List[str]]: Features and labels
        """
        features = []
        labels = []
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return features, labels
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Load and process audio
                        audio = self.load_audio(file_path)
                        combined_features = self.get_combined_features(audio)
                        
                        # Get emotion label
                        emotion = self.get_emotion_from_filename(file)
                        
                        if emotion != 'unknown':
                            features.append(combined_features)
                            labels.append(emotion)
                            
                            logger.info(f"Processed: {file} -> {emotion}")
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
                        continue
        
        logger.info(f"Dataset loaded: {len(features)} samples, {len(set(labels))} emotions")
        return features, labels 