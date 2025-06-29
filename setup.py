#!/usr/bin/env python3
"""
Setup script for Emotion Recognition from Speech project.
This script helps set up the virtual environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_name = "emotion_env"
    
    if os.path.exists(venv_name):
        print(f"⚠️ Virtual environment '{venv_name}' already exists")
        return True
    
    return run_command(f"python -m venv {venv_name}", f"Creating virtual environment '{venv_name}'")

def install_dependencies():
    """Install project dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def create_directories():
    """Create necessary directories."""
    directories = ["model", "ravdess_speech"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"⚠️ Directory already exists: {directory}")

def main():
    """Main setup function."""
    print("🎤 Setting up Emotion Recognition from Speech project...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Determine activation command based on OS
    system = platform.system().lower()
    if system == "windows":
        activate_cmd = "emotion_env\\Scripts\\activate"
    else:
        activate_cmd = "source emotion_env/bin/activate"
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print(f"1. Activate the virtual environment:")
    print(f"   {activate_cmd}")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Download the RAVDESS dataset:")
    print("   - Visit: https://zenodo.org/record/1188976")
    print("   - Extract audio files to 'ravdess_speech/' folder")
    print("\n4. Train the model:")
    print("   python train_model.py")
    print("\n5. Run the web app:")
    print("   streamlit run app.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 