
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os  


load_dotenv()

# Fetch API key from .env
api_key = os.getenv("GOOGLE_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=api_key
)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)