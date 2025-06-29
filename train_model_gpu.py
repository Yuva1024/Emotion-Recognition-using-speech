import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
import pickle
from datetime import datetime

# Import our custom audio processor
from utils.audio_utils import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_gpu():
    """
    Setup GPU configuration for RTX 4060.
    """
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        # Configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Set memory growth for GPU: {gpu}")
        
        # Set mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Enabled mixed precision training")
        
        return True
    else:
        logger.warning("No GPU found. Training will use CPU.")
        return False

class GPUEmotionRecognitionModel:
    """
    GPU-optimized CNN-based emotion recognition model for RTX 4060.
    """
    
    def __init__(self, input_shape=(39, 130), num_classes=6):
        """
        Initialize the GPU-optimized emotion recognition model.
        
        Args:
            input_shape (tuple): Shape of input features (n_mfcc*3, time_frames)
            num_classes (int): Number of emotion classes (6 for RAVDESS)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        self.class_weights = None
        self.gpu_available = setup_gpu()
        
        logger.info(f"Initialized GPUEmotionRecognitionModel with input_shape={input_shape}, num_classes={num_classes}")
        logger.info(f"GPU Available: {self.gpu_available}")
    
    def build_gpu_optimized_model(self):
        """
        Build GPU-optimized CNN model architecture for RTX 4060.
        """
        # Use mixed precision for better GPU performance
        if self.gpu_available:
            dtype = 'float16'
        else:
            dtype = 'float32'
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape, dtype=dtype),
            
            # Reshape for CNN (add channel dimension)
            layers.Reshape((self.input_shape[0], self.input_shape[1], 1)),
            
            # First Convolutional Block - Optimized for GPU
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block (for better feature extraction)
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers with GPU optimization
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax', dtype='float32')  # Output in float32
        ])
        
        # Compile model with GPU-optimized settings
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("GPU-optimized CNN model built successfully")
        model.summary()
        
        return model
    
    def prepare_data_with_balancing(self, features, labels):
        """
        Prepare data with class balancing and GPU-optimized preprocessing.
        
        Args:
            features (list): List of feature arrays
            labels (list): List of emotion labels
            
        Returns:
            tuple: (X, y, class_weight_dict) prepared data
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Convert features to numpy array and pad/truncate to consistent shape
        max_frames = 130
        X_processed = []
        
        for feat in features:
            if feat.shape[1] < max_frames:
                # Pad with zeros
                padded = np.pad(feat, ((0, 0), (0, max_frames - feat.shape[1])), 'constant')
            else:
                # Truncate
                padded = feat[:, :max_frames]
            X_processed.append(padded)
        
        X = np.array(X_processed, dtype=np.float32)
        y = np.array(y_encoded)
        
        # Compute class weights for imbalanced data
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), self.class_weights))
        
        logger.info(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        logger.info(f"Class weights: {class_weight_dict}")
        
        return X, y, class_weight_dict
    
    def train_with_gpu_optimization(self, X, y, class_weight_dict, validation_split=0.2, epochs=150, batch_size=32):
        """
        Train the model with GPU optimization for RTX 4060.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            class_weight_dict (dict): Class weights for balancing
            validation_split (float): Fraction of data for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size optimized for GPU
        """
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        logger.info(f"Using batch size: {batch_size} (optimized for GPU)")
        
        # GPU-optimized callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.7,
                patience=15,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'model/best_gpu_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with GPU optimization
        logger.info("Starting GPU-optimized model training...")
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"GPU-optimized model training completed!")
        logger.info(f"Total training time: {training_time}")
    
    def evaluate_gpu_model(self, X_test, y_test):
        """
        Evaluate the trained GPU model with detailed analysis.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """
        # Predict
        y_pred = self.model.predict(X_test, batch_size=32)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Convert back to original labels
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred_classes)
        
        # Print detailed classification report
        print("\n" + "="*60)
        print("GPU MODEL EVALUATION RESULTS")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original, zero_division=0))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test_original, y_pred_original)
        
        # Plot training history
        self.plot_training_history()
        
        # Print per-class accuracy
        print("\nPer-Class Accuracy:")
        for i, emotion in enumerate(self.label_encoder.classes_):
            mask = y_test == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred_classes[mask] == i)
                print(f"{emotion:10}: {accuracy:.3f} ({np.sum(mask)} samples)")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix with better visualization.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - GPU Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_gpu.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history with better visualization.
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy - GPU Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss - GPU Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_gpu.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_gpu_model(self, model_path='model/emotion_model.h5'):
        """
        Save the GPU-trained model and label encoder.
        
        Args:
            model_path (str): Path to save the model
        """
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save label encoder
        encoder_path = model_path.replace('.h5', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"GPU model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")

def main():
    """
    Main function to train the GPU-optimized emotion recognition model.
    """
    logger.info("Starting GPU-optimized emotion recognition model training...")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(sample_rate=22050, duration=3.0)
    
    # Load dataset
    dataset_path = "ravdess_speech"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist!")
        logger.info("Please download the RAVDESS dataset and place it in the 'ravdess_speech' folder.")
        return
    
    logger.info(f"Loading dataset from {dataset_path}...")
    features, labels = audio_processor.load_dataset(dataset_path)
    
    if len(features) == 0:
        logger.error("No valid audio files found in the dataset!")
        return
    
    # Initialize GPU-optimized model
    model_trainer = GPUEmotionRecognitionModel()
    
    # Build GPU-optimized model
    model_trainer.build_gpu_optimized_model()
    
    # Prepare data with balancing
    X, y, class_weight_dict = model_trainer.prepare_data_with_balancing(features, labels)
    
    # Train model with GPU optimization
    model_trainer.train_with_gpu_optimization(X, y, class_weight_dict, epochs=150, batch_size=32)
    
    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_trainer.evaluate_gpu_model(X_test, y_test)
    
    # Save GPU model
    model_trainer.save_gpu_model()
    
    logger.info("GPU-optimized training completed successfully!")

if __name__ == "__main__":
    main()
