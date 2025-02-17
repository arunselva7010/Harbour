import streamlit as st
from streamlit.components.v1 import html

# Custom HTML/CSS/JavaScript for the audio recorder
custom_recorder_code = """
<div style="text-align: center; margin-top: 20px;">
    <button id="recordButton" style="background-color: #6200ea; border: none; border-radius: 50%; width: 80px; height: 80px; cursor: pointer;">
        <img src="https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif" style="width: 100%; height: 100%; border-radius: 50%;" />
    </button>
    <p id="recordingStatus" style="margin-top: 10px; color: #6200ea;">Click the mic to start recording</p>
    <audio id="audioPlayback" controls style="margin-top: 20px; display: none;"></audio>
</div>

<script>
    let chunks = [];
    let recorder;
    let audioBlob;

    document.getElementById('recordButton').addEventListener('click', async () => {
        if (recorder && recorder.state === 'recording') {
            recorder.stop();
            document.getElementById('recordingStatus').innerText = 'Recording stopped';
        } else {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorder = new MediaRecorder(stream);
            
            recorder.ondataavailable = event => {
                chunks.push(event.data);
            };
            
            recorder.onstop = () => {
                audioBlob = new Blob(chunks, { type: 'audio/wav' });
                chunks = [];
                const audioURL = window.URL.createObjectURL(audioBlob);
                const audio = document.getElementById('audioPlayback');
                audio.src = audioURL;
                audio.style.display = 'block';
                document.getElementById('recordingStatus').innerText = 'Recording complete';
            };
            
            recorder.start();
            document.getElementById('recordingStatus').innerText = 'Recording...';
        }
    });
</script>
"""

# Display the custom recorder
html(custom_recorder_code, height=300)

# Placeholder for processing or saving the audio (to be implemented)
if st.button("Save Recording"):
    st.write("Audio recording would be processed or saved here.")
