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

class SimpleEmotionRecognitionModel:
    """
    Simple but effective emotion recognition model to avoid model collapse.
    """
    
    def __init__(self, input_shape=(39, 130), num_classes=6):
        """
        Initialize the simple emotion recognition model.
        
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
        
        logger.info(f"Initialized SimpleEmotionRecognitionModel with input_shape={input_shape}, num_classes={num_classes}")
    
    def build_simple_model(self):
        """
        Build a simple but effective model architecture.
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Flatten the input
            layers.Flatten(),
            
            # Simple dense layers with moderate regularization
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with conservative settings
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,  # Higher learning rate for simple model
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
        logger.info("Simple model built successfully")
        model.summary()
        
        return model
    
    def prepare_data(self, features, labels):
        """
        Prepare data with proper normalization and class balancing.
        
        Args:
            features (list): List of feature arrays
            labels (list): List of emotion labels
            
        Returns:
            tuple: (X, y, class_weight_dict) prepared data
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Convert features to numpy array
        X = np.array(features, dtype=np.float32)
        y = np.array(y_encoded)
        
        # Additional normalization per sample
        for i in range(X.shape[0]):
            sample = X[i]
            # Normalize each sample to have zero mean and unit variance
            sample = (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
            X[i] = sample
        
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
    
    def train_model(self, X, y, class_weight_dict, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the simple model with conservative settings.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            class_weight_dict (dict): Class weights for balancing
            validation_split (float): Fraction of data for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Conservative callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'model/best_simple_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting simple model training...")
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
        
        logger.info(f"Simple model training completed!")
        logger.info(f"Total training time: {training_time}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model with detailed analysis.
        
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
        print("SIMPLE MODEL EVALUATION RESULTS")
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
        
        # Check for model collapse
        unique_predictions = np.unique(y_pred_classes)
        print(f"\nUnique predictions: {unique_predictions}")
        if len(unique_predictions) == 1:
            print("WARNING: Model is predicting only one class!")
            predicted_class = self.label_encoder.inverse_transform([unique_predictions[0]])[0]
            print(f"Model is only predicting: {predicted_class}")
        else:
            print(f"Model is predicting {len(unique_predictions)} different classes - good!")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Simple Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy - Simple Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss - Simple Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='model/emotion_model_simple.h5'):
        """
        Save the trained model and label encoder.
        
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
        
        logger.info(f"Simple model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")

def main():
    """
    Main function to train the simple emotion recognition model.
    """
    logger.info("Starting simple emotion recognition model training...")
    
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
    
    # Initialize simple model
    model_trainer = SimpleEmotionRecognitionModel()
    
    # Build simple model
    model_trainer.build_simple_model()
    
    # Prepare data
    X, y, class_weight_dict = model_trainer.prepare_data(features, labels)
    
    # Train model
    model_trainer.train_model(X, y, class_weight_dict, epochs=100, batch_size=32)
    
    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_trainer.evaluate_model(X_test, y_test)
    
    # Save model
    model_trainer.save_model()
    
    logger.info("Simple training completed successfully!")

if __name__ == "__main__":
    main() 