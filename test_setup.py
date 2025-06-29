#!/usr/bin/env python3
"""
Test script to verify the Emotion Recognition project setup.
This script checks if all dependencies are installed and the project is ready to run.
"""

import sys
import importlib
import os

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        if package_name:
            importlib.import_module(package_name)
        else:
            importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def test_file_exists(file_path, description):
    """Test if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (not found)")
        return False

def test_directory_exists(dir_path, description):
    """Test if a directory exists."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}: {dir_path} (not found)")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Emotion Recognition project setup...")
    print("=" * 60)
    
    # Test Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} (3.8+ required)")
        return False
    
    print("\n📦 Testing required packages:")
    
    # Test core dependencies
    core_deps = [
        ("tensorflow", "tensorflow"),
        ("keras", "tensorflow.keras"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("librosa", "librosa"),
        ("streamlit", "streamlit"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("soundfile", "soundfile"),
        ("pydub", "pydub")
    ]
    
    all_core_ok = True
    for module_name, package_name in core_deps:
        if not test_import(module_name, package_name):
            all_core_ok = False
    
    print("\n📁 Testing project structure:")
    
    # Test project files
    project_files = [
        ("app.py", "Streamlit application"),
        ("train_model.py", "Model training script"),
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Project documentation"),
        ("setup.py", "Setup script"),
        ("utils/audio_utils.py", "Audio utilities module")
    ]
    
    all_files_ok = True
    for file_path, description in project_files:
        if not test_file_exists(file_path, description):
            all_files_ok = False
    
    # Test directories
    project_dirs = [
        ("utils/", "Utils directory"),
        ("model/", "Model directory"),
        ("ravdess_speech/", "Dataset directory")
    ]
    
    all_dirs_ok = True
    for dir_path, description in project_dirs:
        if not test_directory_exists(dir_path, description):
            all_dirs_ok = False
    
    print("\n🔧 Testing custom modules:")
    
    # Test custom modules
    try:
        from utils.audio_utils import AudioProcessor
        print("✅ AudioProcessor class imported successfully")
        custom_modules_ok = True
    except ImportError as e:
        print(f"❌ AudioProcessor import failed: {e}")
        custom_modules_ok = False
    
    print("\n" + "=" * 60)
    
    # Summary
    if all_core_ok and all_files_ok and all_dirs_ok and custom_modules_ok:
        print("🎉 All tests passed! The project is ready to use.")
        print("\n📋 Next steps:")
        print("1. Download RAVDESS dataset to 'ravdess_speech/' folder")
        print("2. Run: python train_model.py")
        print("3. Run: streamlit run app.py")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("\n🔧 To fix:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Ensure all project files are present")
        print("3. Check Python version (3.8+ required)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 