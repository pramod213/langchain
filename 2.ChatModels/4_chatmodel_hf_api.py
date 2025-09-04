from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("Write five lines on India")

print(result.content)