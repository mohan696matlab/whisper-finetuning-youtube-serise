import streamlit as st
import pyaudiowpatch as pyaudio
import torch
import numpy as np
import os
import time
from datetime import datetime
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration



# Initialize PyAudio
p = pyaudio.PyAudio()


# Function to list all input devices
def list_audio_devices():
    device_list = []
    for idx in range(p.get_device_count()):
        device_list.append(p.get_device_info_by_index(idx)['name'])
    return device_list

# Function to cache the model and processor
@st.cache_resource          
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.load_state_dict(torch.load('finetuned_on_odia_steps_1000.pth', weights_only=True))# loading the model
    model.to('cuda')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model

# Load Whisper model and processor
processor, model = load_whisper_model()

# Function to process audio and generate transcription
def transcribe_audio(audio_chunk, processor, model, language, task):
    input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to('cuda'), language=language, task=task)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Initialize session state to store transcriptions
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = ""

# Function to append transcription to session state
def update_transcription(new_text,time_taken):
    st.session_state["transcriptions"] += f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {new_text}----{time_taken:.2f}s\n"


# Streamlit UI
st.title(":studio_microphone: Live Audio Transcription App")

# Allow user to select input device
devices = list_audio_devices()
input_device = st.selectbox("Select Microphone", devices)

# Select language and task
language = 'bengali'
task = 'transcribe'


# Add slider for TRANSCRIPTION_INTERVAL (max 30, default 20)
TRANSCRIPTION_INTERVAL = st.slider("Set Transcription Interval (in seconds)", min_value=5, max_value=30, value=20)


# Create button states
start_button = st.button('Start Transcription')
stop_button = st.button('Stop Transcription')



if start_button:
    input_device_index = devices.index(input_device)

    # Settings for recording audio
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = int(p.get_device_info_by_index(input_device_index)['defaultSampleRate'])
    WHISPER_RATE = 16000  # Whisper expects 16kHz input
    # TRANSCRIPTION_INTERVAL = 30  # Set a short interval for responsiveness

    # Open a stream to record audio
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=input_device_index)

    st.write(f"Listening for audio from **{input_device}**... Speak now. :studio_microphone:")
    st.write(f"Transcription interval is **{TRANSCRIPTION_INTERVAL}s** :hourglass_flowing_sand:")

    live_text = st.empty()  # Placeholder for live transcription text
    st.session_state["transcriptions"] = ""
    
    audio_frames = np.array([], dtype=np.float32)

    # Start recording and transcribing
    stop_recording = False
    while not stop_recording:
        if stop_button:
            stop_recording = True
            break

        # Read a chunk of audio
        data = stream.read(RATE * TRANSCRIPTION_INTERVAL, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * WHISPER_RATE / RATE))  # Resample to 16kHz
        
        # Append to audio frames for future use (e.g., saving audio)
        audio_frames = np.append(audio_frames, audio_chunk)
        
        task_start = time.time()
        # Get the transcription
        transcription_text = transcribe_audio(audio_chunk, processor, model, language=language, task=task)
        time_taken = time.time() - task_start
        # Update live transcription
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_transcription(transcription_text,time_taken)  # Save transcription to session state
        live_text.markdown(f"{st.session_state['transcriptions']}")

        # Optionally save the audio file
        file_path = "transcriptions.txt"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {transcription_text}\n")
        
        
        # Add a short 

    if stop_recording:
        st.write("Stopped listening.")
        stream.stop_stream()
        stream.close()
        p.terminate()


