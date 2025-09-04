from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from langchain_google_genai import GoogleGenerativeAIEmbeddings # pyright: ignore[reportMissingImports]
import os

# Load environment variables (make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions=32 , # Gemini embedding model
    google_api_key=api_key
)

documents = [
    "Delhi is the capital of Indaia",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embedding for a query
result = embedding.embed_documents(documents)

print(str(result[:32]))   # print first 32 dimensions
print(len(result))        # embedding length (usually 768)
