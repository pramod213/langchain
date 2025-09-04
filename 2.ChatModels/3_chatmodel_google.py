from langchain_google_genai import ChatGoogleGenerativeAI # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

result= model.invoke("Write a 5 line poem on Cricket")

print(result.content)