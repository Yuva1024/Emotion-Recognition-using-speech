#!/usr/bin/env python3
"""
Test script to verify deployment dependencies work correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    try:
        print("Testing imports...")
        
        # Core packages
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        # Machine learning packages
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        import keras
        print(f"✅ Keras {keras.__version__} imported successfully")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
        
        # Audio processing
        import librosa
        print("✅ Librosa imported successfully")
        
        import soundfile as sf
        print("✅ Soundfile imported successfully")
        
        # Visualization
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import plotly.express as px
        print("✅ Plotly imported successfully")
        
        # Image processing
        from PIL import Image
        print("✅ Pillow imported successfully")
        
        print("\n🎉 All imports successful! Deployment should work.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_loading():
    """Test that model files can be loaded."""
    try:
        import os
        import pickle
        from tensorflow import keras
        
        print("\nTesting model loading...")
        
        # Check if model files exist
        model_files = [
            'model/best_gpu_model.h5',
            'model/best_improved_model.h5',
            'model/best_simple_model.h5',
            'model/emotion_model_simple.h5',
            'model/emotion_model.h5'
        ]
        
        found_models = []
        for model_file in model_files:
            if os.path.exists(model_file):
                found_models.append(model_file)
                print(f"✅ Found model: {model_file}")
        
        if not found_models:
            print("❌ No model files found!")
            return False
        
        # Try to load the first available model
        model_path = found_models[0]
        model = keras.models.load_model(model_path)
        print(f"✅ Successfully loaded model: {model_path}")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Check for encoder
        encoder_path = model_path.replace('.h5', '_encoder.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            print(f"✅ Successfully loaded encoder: {encoder_path}")
            print(f"   Classes: {encoder.classes_}")
        else:
            print("⚠️  No encoder file found")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_audio_utils():
    """Test that audio utilities can be imported."""
    try:
        print("\nTesting audio utilities...")
        
        from utils.audio_utils import AudioProcessor
        print("✅ AudioProcessor imported successfully")
        
        # Test creating an instance
        processor = AudioProcessor()
        print("✅ AudioProcessor instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio utilities error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing deployment setup...\n")
    
    # Run all tests
    tests = [
        test_imports,
        test_model_loading,
        test_audio_utils
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    if all(results):
        print("🎉 All tests passed! Your deployment should work correctly.")
        print("\nNext steps:")
        print("1. Commit and push your changes to GitHub")
        print("2. Deploy to Streamlit Cloud using streamlit_app.py")
        print("3. Use requirements_streamlit.txt for better compatibility")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        print("\nFailed tests:")
        test_names = ["Imports", "Model Loading", "Audio Utils"]
        for i, (name, result) in enumerate(zip(test_names, results)):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {name}: {status}")
    
    print("\n" + "="*50) 