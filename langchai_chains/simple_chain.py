from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt | model | parser  # LCEL langchain Expression Language

result = chain.invoke({'topic': 'cricket'})

print(result)