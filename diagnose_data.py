import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
from datetime import datetime

# Import our custom audio processor
from utils.audio_utils import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset():
    """
    Analyze the dataset to identify potential issues.
    """
    logger.info("Starting dataset analysis...")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(sample_rate=22050, duration=3.0)
    
    # Load dataset
    dataset_path = "ravdess_speech"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist!")
        return
    
    logger.info(f"Loading dataset from {dataset_path}...")
    features, labels = audio_processor.load_dataset(dataset_path)
    
    if len(features) == 0:
        logger.error("No valid audio files found in the dataset!")
        return
    
    # Analyze class distribution
    label_counts = Counter(labels)
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"\nTotal samples: {len(features)}")
    print(f"Number of emotions: {len(label_counts)}")
    print("\nClass distribution:")
    for emotion, count in sorted(label_counts.items()):
        percentage = (count / len(features)) * 100
        print(f"  {emotion:10}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Analyze feature statistics
    print(f"\nFeature analysis:")
    print(f"  Number of features per sample: {len(features)}")
    print(f"  Feature shape: {features[0].shape}")
    
    # Convert to numpy array for analysis
    X = np.array(features)
    print(f"  Full dataset shape: {X.shape}")
    print(f"  Feature statistics:")
    print(f"    Mean: {np.mean(X):.6f}")
    print(f"    Std:  {np.std(X):.6f}")
    print(f"    Min:  {np.min(X):.6f}")
    print(f"    Max:  {np.max(X):.6f}")
    
    # Check for NaN or infinite values
    print(f"  NaN values: {np.isnan(X).sum()}")
    print(f"  Infinite values: {np.isinf(X).sum()}")
    
    # Analyze feature variance per class
    print(f"\nFeature variance per class:")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    for i, emotion in enumerate(label_encoder.classes_):
        mask = y_encoded == i
        if np.sum(mask) > 0:
            class_features = X[mask]
            variance = np.var(class_features)
            print(f"  {emotion:10}: variance = {variance:.6f}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    emotions = list(label_counts.keys())
    counts = list(label_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
    plt.bar(emotions, counts, color=colors)
    plt.title('Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Plot feature distribution
    plt.subplot(1, 2, 2)
    plt.hist(X.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test simple model to see if it can learn
    print(f"\nTesting simple model...")
    test_simple_model(X, y_encoded, label_encoder)

def test_simple_model(X, y, label_encoder):
    """
    Test a very simple model to see if the data is learnable.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Build a very simple model
    model = models.Sequential([
        layers.Input(shape=X.shape[1:]),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for a few epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print per-class accuracy
    print(f"\nSimple model per-class accuracy:")
    for i, emotion in enumerate(label_encoder.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            accuracy = np.mean(y_pred_classes[mask] == i)
            print(f"  {emotion:10}: {accuracy:.3f} ({np.sum(mask)} samples)")
    
    # Check if model is predicting only one class
    unique_predictions = np.unique(y_pred_classes)
    print(f"\nUnique predictions: {unique_predictions}")
    if len(unique_predictions) == 1:
        print("WARNING: Model is predicting only one class!")
        predicted_class = label_encoder.inverse_transform([unique_predictions[0]])[0]
        print(f"Model is only predicting: {predicted_class}")
    else:
        print("Model is predicting multiple classes - data seems learnable")

if __name__ == "__main__":
    analyze_dataset() 