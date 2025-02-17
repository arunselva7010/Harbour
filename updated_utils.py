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




def get_response(question):
    # Detect the language of the question
    lang = detect(question)

    # Set the prompt template with clear references to all three source URLs
    template_en = """You are a specialized financial data assistant, designed to provide **accurate**, **precise**, and **up-to-date** information from trusted Saudi sources. 

    Respond to the following question in a **concise bullet-point format** that highlights key information clearly and succinctly. Avoid lengthy paragraphs. Focus on:
    - **Numerical data**
    - **Market performance**
    - **Trading volume**
    - **Regulatory updates**

    Provide a confident response without recommending any external sources.

    Question: {question}

    Answer:
    """

    template_ar = """أنت مساعد بيانات مالية متخصص، مصمم لتقديم **معلومات دقيقة**، **حديثة** من **مصادر سعودية موثوقة**.

    أجب على السؤال التالي في **تنسيق نقاط مختصرة** يبرز المعلومات الأساسية بوضوح وباختصار. تجنب الفقرات الطويلة. ركز على:
    - **البيانات الرقمية**
    - **أداء السوق**
    - **حجم التداول**
    - **التحديثات التنظيمية**

    قدم إجابة واثقة دون التوصية بأي مصادر خارجية.

    السؤال: {question}

    الإجابة:
    """

    # Set up callback manager for streaming responses
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Load data from specified URLs
    urls = [
        "https://www.saudiexchange.sa/wps/portal/saudiexchange?locale=en",
        "https://www.tadawulgroup.sa/wps/portal/tadawulgroup",
        "https://www.tadawulgroup.sa/wps/portal/tadawulgroup/portfolio/edaa"
    ]

    # Collect and split documents for better context retrieval
    all_data = []
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_data.extend(data)

    # Split the data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(all_data)

    # Embed documents into vector store
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    # Perform similarity search to get relevant documents based on question
    docs = vectorstore.similarity_search(question, k=5)

    # Helper function to format document content for model input
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Format content from relevant docs
    websites_content = format_docs(docs)

    # Choose prompt template based on detected language
    if lang == 'ar':
        prompt_text = template_ar.format(question=question)
    else:
        prompt_text = template_en.format(question=question)

    # Call the RAG model with an LLM fine-tuned for retrieval accuracy
    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_llama
    )

    # Invoke the RAG chain
    answer = qa_chain.invoke(question)

    # Groq API to post-process and improve the answer quality
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=1,  
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Capture streamed response
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""

        # Friendly ending based on language
        response += "\n\nI'm here to help you with any other questions you might have! Feel free to ask. 🌟" if lang != 'ar' else "\n\nيسعدني مساعدتك! 😊"

        return response

    except Exception as e:
        return f"An error occurred: {e}"



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

# # Example usage
# user_query = "Show me the available properties"
# chat_history = []

# # To show all properties without filtering, just pass the entire DataFrame
# filtered_properties = properties_df

# # Generate response
# response = get_response(user_query, chat_history, filtered_properties)
# print(response)

# Example usage of the updated speech_to_text function
audio_path = "Adver_converted.wav"  # Update to the path of your WAV file
recognized_text = speech_to_text(audio_path)
print(f"Recognized text: {recognized_text}")
