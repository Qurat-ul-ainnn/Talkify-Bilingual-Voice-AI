from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from gtts.lang import tts_langs
import streamlit as st
import os
from dotenv import load_dotenv  # For environment variable management
from langdetect import detect  # Language detection library

# Load environment variables
load_dotenv()

# Retrieve the Google API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found! Make sure you have a .env file with GOOGLE_API_KEY.")

# Set page configuration
st.set_page_config(page_title="AI Voice Assistant", page_icon="🤖")

# App title and description
st.title("AI Voice Assistant 🎙️")
st.subheader("Interact in English or Urdu with Real-Time Voice Input")

# User preferences for language selection
input_language = st.selectbox("Select Input Language:", ["English", "Urdu"], index=1)
output_language = st.selectbox("Select Output Language:", ["English", "Urdu"], index=0)

# Language mapping for speech recognition and text-to-speech
lang_map = {"English": "en", "Urdu": "ur"}

# Function to detect language from the input text
def detect_input_language(text):
    try:
        # Detect the language of the input text
        detected_lang = detect(text)
        return "en" if detected_lang == "en" else "ur"
    except Exception as e:
        st.error(f"Error detecting language: {str(e)}")
        return "en"  # Default to English if detection fails

# Chat prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            f"You are a helpful AI assistant. Please respond to the user queries in {output_language} language."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Initialize chat message history
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Load the AI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the chain
chain = prompt | model | StrOutputParser()

# Add history handling
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Supported languages for text-to-speech
langs = tts_langs().keys()

# Instructions
st.write("Press the button and start speaking:")

# Process voice input
with st.spinner(f"Converting Speech To Text in {input_language}..."):
    speech_lang = lang_map[input_language]
    text = speech_to_text(language=speech_lang, use_container_width=True, just_once=True, key="STT")

if text:
    st.chat_message("human").write(text)

    # Detect the language of the input (this step is to detect language dynamically if not selected)
    detected_language = detect_input_language(text)
    input_language = "English" if detected_language == "en" else "Urdu"  # Adjust the input language

    # Generate the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = chain_with_history.stream({"question": text}, {"configurable": {"session_id": "any"}})

        for res in response:
            full_response += res or ""
            message_placeholder.markdown(full_response)

    # Convert the response to speech in the selected output language
    with st.spinner(f"Converting Text To Speech in {output_language}..."):
        tts_lang = lang_map[output_language]
        tts = gTTS(text=full_response, lang=tts_lang)
        tts.save("output.mp3")
        st.audio("output.mp3")
else:
    st.warning("Please press the button and start speaking.")
