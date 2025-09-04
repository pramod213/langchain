

from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from langchain_google_genai import GoogleGenerativeAIEmbeddings # pyright: ignore[reportMissingImports]
from sklearn.metrics.pairwise import cosine_similarity # pyright: ignore[reportMissingModuleSource]
import numpy as np
import os

# Loads API key from .env (must contain GOOGLE_API_KEY)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initializing Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",   # Gemini embedding model
    google_api_key=api_key
)

# Documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Rohit Sharma is known for his elegent batting and record-breaking double centuries.",
    "Sachin Tendulkar , also known as the 'God of Cricket', holds many batting records.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = "Tell me about jasprit bumrah"

# Create embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Cosine similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

index , score = sorted(list(enumerate(similarities)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is : ", score)
