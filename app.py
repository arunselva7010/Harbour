import streamlit as st
import os
import uuid
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from utils import speech_to_text, get_response, filter_properties

# Initialize the pipeline for speech recognition
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I am 𝖪𝖦𝖨𝖲𝖫 𝖯𝗋𝗈𝗉𝖾𝗋.𝖠𝖨🏡. Please tell me the area you're interested in, the property type you're looking for, and your budget. I'm here to help you find the perfect property!"}
        ]
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "properties" not in st.session_state:
        st.session_state.properties = filter_properties()  # Initial load of properties
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

initialize_session_state()

st.header("𝖪𝖦𝖨𝖲𝖫 𝖯𝗋𝗈𝗉𝖾𝗋.𝖠𝖨🏡")

# Create a container for the footer
footer_container = st.container()

# Initialize the user's query to None
user_query = None

# Handle audio input in the footer container
with footer_container:
    transcript = None

        
        # You need to handle the actual audio recording functionality separately here.
    audio_bytes = audio_recorder(text=None, icon_size="3x",  sample_rate=16000)

    if audio_bytes:
        # Write the audio bytes to a file
        with st.spinner("...."):
            webm_file_path = "temp_audio.wav"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)
            transcript = speech_to_text(webm_file_path)
            os.remove(webm_file_path)
            user_query = transcript

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If there's a user query, process it
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.user_query = user_query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

# Generate and display the assistant's response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading data...."):
            if st.session_state.user_query:
                available_properties = filter_properties()
                final_response = get_response(st.session_state.user_query, st.session_state.chat_history, available_properties)
            else:
                final_response = "How can I assist you with real estate today?"
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

footer_container.float("bottom: 0rem; right: 10px;")

