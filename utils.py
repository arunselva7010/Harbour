import os
import pandas as pd
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
import streamlit as st
from groq import Groq

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Load environment variables

groq_api_key = "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"

# Initialize the Groq client
client = Groq(api_key=groq_api_key)

# Load the CSV file
properties_df = pd.read_csv("properties.csv")

def filter_properties(location=None, property_type=None, budget=None):
    filtered_df = properties_df

    if location:
        filtered_df = filtered_df[filtered_df['location'].str.contains(location, case=False)]

    if property_type:
        filtered_df = filtered_df[filtered_df['property_type'].str.contains(property_type, case=False)]

    if budget:
        filtered_df = filtered_df[filtered_df['price'] <= budget]

    return filtered_df

def format_properties_text(properties_df):
    if properties_df.empty:
        return "It seems there aren't any properties matching your criteria at the moment. Maybe we can adjust your preferences?"

    properties_text = []
    for _, row in properties_df.iterrows():
        property_details = (
            f"**Location:** {row['location']} , **Property Type:** {row['property_type']} , "
            f"**Price:** {row['price']} , "
            f"**Bedrooms:** {row['no_of_bedrooms']} , **Bathrooms:** {row['no_of_bathrooms']} , "
            f"**Amenities:** {row['amenities']}"
        )
        properties_text.append(property_details)

    return "\n\n".join(properties_text)

def get_response(user_query, chat_history, available_properties):
    properties_text = format_properties_text(available_properties)

    prompt_template = f"""
    You are a friendly and professional real estate assistant named KGISL'S Property.AI🏡. Your main goal is to help users find their ideal property by providing detailed, accurate, and conversational responses. Always strive to make the interaction feel natural and engaging.

    Instructions:

    1. **User Query Interpretation**:
        - Carefully understand the user's request, including any specified location, property type, budget, and additional preferences.
        - If the user’s query is vague or incomplete, politely ask for more details to better assist them.

    2. **Property Information**:
        - Filter properties by the user’s specified criteria (location, property type, budget, etc.).
        - Present the results using clear, engaging, and friendly language.
        - Introduce the results with a welcoming phrase like "Sure, I’ve found some beautiful properties for you in [location]! Here are the details:".
        - Don't start with "Hi there! I am KGISL'S Property.AI" every time; instead, respond naturally based on the user's query.
        - List properties in bullet points or numbered format for easy readability.
        - Use descriptive and positive language (e.g., "beautiful", "spacious", "modern", "luxurious") to make the properties sound appealing.
        - **Handle N/A Values**: If a property type does not require "Bedrooms" or "Bathrooms" (e.g., office spaces), either omit these fields or replace them with a friendly message like "Not applicable for this property type."

    3. **Response Formatting**:
        - When a user requests property information:
            - Filter properties by location, property type, and budget if specified.
            - Present the results in a bullet-point format with key property details, using bold text to highlight important information.
            - Ensure that the information is visually clear and appealing.
            - Start each property detail with a bullet point (•) and ensure there is space between each point, like this:
            
                • **Location:** Palm Jumeirah, **Property Type:** Apartment, **Price:** 1,950,000 AED, **Bedrooms:** 1, **Bathrooms:** 2, **Amenities:** Pool, Gym, Parking

                • **Location:** Downtown Dubai, **Property Type:** Villa, **Price:** 1,700,000 AED, **Bedrooms:** 2, **Bathrooms:** 3, **Amenities:** Pool, Gym, Parking

            - Ensure that each property is separated by a blank line to maintain clarity and avoid cluttered paragraphs.
            - If the list of properties is extensive, break the response into multiple parts to ensure the response is not cut off.

    4. **Exception Handling**:
        - If no properties match the user’s criteria, respond warmly and suggest alternatives: "It seems there are no properties that match your criteria at the moment. How about we adjust your search?"
        - If the user's request is unclear or incomplete, respond with: "Could you please provide more details? For example, the location or property type you're interested in."

    5. **Engaging Interaction**:
        - Keep the tone friendly, conversational, and professional. For instance: "I’m excited to show you these options!" or "Let’s find the perfect place for you!"
        - Offer to assist with further queries or modifications to their search: "Would you like to refine your search or explore these options in more detail?"

    6. **Error Handling**:
        - If there’s an error in retrieving data or any other issue, respond apologetically and offer to try again: "I’m sorry, something went wrong in retrieving the properties. Let me try again."

    7. **General Tone**:
        - Maintain a warm and inviting tone throughout. Use positive reinforcement and show eagerness to help the user find the best property.

    8. **Emojis**:
        - Use emojis to add a touch of friendliness and personality to your responses.

    9. **Complete Response**:
        - Ensure that your response is complete and includes all necessary information. Avoid stopping mid-response; if a response is lengthy, consider breaking it into multiple parts to ensure completeness. Always conclude your response in a way that feels natural and finished.

        





    Available properties:
    {properties_text}

    Chat history:
    {chat_history}

    User question:
    {user_query}
    """


    # Use Groq's chat completion endpoint with the LLaMA model
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": prompt_template}],
        temperature=0.7,  
        max_tokens=2000,  
        top_p=1,
        stream=False,  # Set to False if you want the entire response at once
    )

    # Retrieve the response
    response_text = completion.choices[0].message.content.strip()

    return response_text

# Example usage:
# user_query = "I'm looking for a 2-bedroom apartment in Downtown Dubai under 1 million."
# chat_history = "..."
# available_properties = filter_properties(location="Downtown Dubai", property_type="Apartment", budget=1000000)
# response = get_response(user_query, chat_history, available_properties)
# print(response)


import os
from groq import Groq
from langdetect import detect, DetectorFactory
from pydub import AudioSegment
from pydub.effects import normalize
import io

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Initialize the Groq client with your API key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

def preprocess_audio(audio_path):
    """
    Preprocess the audio by normalizing volume and reducing background noise.
    """
    audio = AudioSegment.from_file(audio_path)
    normalized_audio = normalize(audio)
    
    # Optional: Apply other noise reduction techniques here
    
    buffer = io.BytesIO()
    normalized_audio.export(buffer, format="wav")
    buffer.seek(0)
    
    return buffer

def speech_to_text(audio_path):
    filename = os.path.basename(audio_path)
    
    # Preprocess the audio to improve quality
    processed_audio = preprocess_audio(audio_path)
    
    # Transcribe the audio using Groq's Whisper model
    transcription = client.audio.transcriptions.create(
        file=(filename, processed_audio.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    
    # Print the transcription object to understand its structure
    print(transcription)

    # Extract the recognized text from the transcription response
    try:
        recognized_text = transcription.text
    except AttributeError:
        print("Failed to extract text from the transcription response.")
        return ""

    # Optionally, detect the language if needed
    try:
        detected_language = detect(recognized_text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        detected_language = 'unknown'

    # Print the recognized text and detected language for debugging purposes
    print(f"Detected language: {detected_language}")
    print(f"User said: {recognized_text}")

    # Check if the detected language is English
    if detected_language == 'en':
        return recognized_text
    else:
        print("Non-English text detected. Returning empty string.")
        return ""

# Example usage of the updated speech_to_text function
audio_path = "Adver_converted.wav"  # Update to the path of your WAV file
recognized_text = speech_to_text(audio_path)
print(f"Recognized text: {recognized_text}")


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# Example usage
user_query = "Show me the available properties"
chat_history = []

# To show all properties without filtering, just pass the entire DataFrame
filtered_properties = properties_df

# Generate response
response = get_response(user_query, chat_history, filtered_properties)
print(response)

# Example usage of the updated speech_to_text function
audio_path = "Adver_converted.wav"  # Update to the path of your WAV file
recognized_text = speech_to_text(audio_path)
print(f"Recognized text: {recognized_text}")
