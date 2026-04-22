from modules.install_voice_deps import install_packages
install_packages()

import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import speech_to_text

def normalize_voice_query(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    
    # remove filler words
    fillers = ["uh", "um", "like", "please", "can you"]
    for f in fillers:
        text = text.replace(f, "")
        
    # normalize course codes
    text = text.replace("cs two zero two", "cs-202")
    text = text.replace("cs 202", "cs-202")
    
    return text.strip()

def get_voice_input():
    """
    Returns the transcribed text from the voice input.
    Uses browser-based Web Speech API via streamlit-mic-recorder.
    """
    text = speech_to_text(
        language='en', 
        use_container_width=True, 
        just_once=True, 
        key='STT'
    )
    return text

def get_voice_input_fallback():
    """
    Fallback method using SpeechRecognition library directly 
    (uses server microphone, primarily for local development).
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source, timeout=5)
    try:
        st.info("Processing...")
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
