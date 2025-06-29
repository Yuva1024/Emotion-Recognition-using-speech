import os
import numpy as np
import pickle
from tensorflow import keras
from utils.audio_utils import AudioProcessor

def test_model():
    """
    Test the trained model with a sample from the dataset.
    """
    print("🧪 Testing Emotion Recognition Model")
    print("=" * 50)
    
    # Load model
    model_path = 'model/emotion_model_simple.h5'
    encoder_path = 'model/emotion_model_simple_encoder.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return
    
    try:
        # Load model and encoder
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model classes: {list(label_encoder.classes_)}")
        
        # Load audio processor
        audio_processor = AudioProcessor(sample_rate=22050, duration=3.0)
        
        # Find a sample audio file
        dataset_path = "ravdess_speech"
        sample_file = None
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    sample_file = os.path.join(root, file)
                    break
            if sample_file:
                break
        
        if not sample_file:
            print("❌ No sample audio file found!")
            return
        
        print(f"🎵 Testing with: {os.path.basename(sample_file)}")
        
        # Process audio
        audio = audio_processor.load_audio(sample_file)
        features = audio_processor.get_combined_features(audio)
        
        # Get expected emotion from filename
        expected_emotion = audio_processor.get_emotion_from_filename(os.path.basename(sample_file))
        print(f"🎯 Expected emotion: {expected_emotion}")
        
        # Normalize features (same as training)
        features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        features_reshaped = features_normalized.reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(features_reshaped, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[0][predicted_class_idx]
        
        print(f"\n📊 Prediction Results:")
        print(f"   Predicted: {predicted_emotion}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Correct: {'✅' if predicted_emotion == expected_emotion else '❌'}")
        
        # Show all probabilities
        print(f"\n📈 All Emotion Probabilities:")
        for i, emotion in enumerate(label_encoder.classes_):
            prob = predictions[0][i]
            marker = "🎯" if emotion == predicted_emotion else "  "
            print(f"   {marker} {emotion:10}: {prob:.1%}")
        
        # Test with different emotions
        print(f"\n🧪 Testing Multiple Samples:")
        test_count = 0
        correct_count = 0
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files[:10]:  # Test first 10 files
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    expected = audio_processor.get_emotion_from_filename(file)
                    
                    # Process audio
                    audio = audio_processor.load_audio(file_path)
                    features = audio_processor.get_combined_features(audio)
                    features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
                    features_reshaped = features_normalized.reshape(1, -1)
                    
                    # Predict
                    predictions = model.predict(features_reshaped, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted = label_encoder.inverse_transform([predicted_class_idx])[0]
                    
                    test_count += 1
                    if predicted == expected:
                        correct_count += 1
                    
                    print(f"   {file}: Expected={expected}, Predicted={predicted} {'✅' if predicted == expected else '❌'}")
        
        accuracy = correct_count / test_count if test_count > 0 else 0
        print(f"\n📊 Test Results: {correct_count}/{test_count} correct ({accuracy:.1%})")
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")

if __name__ == "__main__":
    test_model() 