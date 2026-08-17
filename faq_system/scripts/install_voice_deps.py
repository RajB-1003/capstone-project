import subprocess
import sys

def install_packages():
    try:
        import speech_recognition
        import streamlit_mic_recorder
    except ImportError:
        print("Installing speech recognition packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SpeechRecognition", "streamlit-mic-recorder"])

install_packages()
