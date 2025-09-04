from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-oss-120B', temperature = 0, max_completion_tokens=10)

result= model.invoke("What is the capital of India?")

print(result.content)