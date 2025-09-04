from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables (make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # use "pro" if you have quota
    api_key=os.getenv("GOOGLE_API_KEY")
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button('Summarize') and user_input:
    result = model.invoke(user_input)

    # Safely handle the response
    if isinstance(result.content, list) and len(result.content) > 0:
        st.write(result.content[0].text)
    else:
        st.write(result.content)
