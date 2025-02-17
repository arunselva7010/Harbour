import librosa
from transformers import pipeline

audio_file = "Adver_converted.wav"
audio, sr = librosa.load(audio_file, sr=16000)  # Load with a sample rate of 16kHz

# Whisper ASR pipeline
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

# Pass the audio data directly to the pipeline
result = transcriber(audio)

print(result)
