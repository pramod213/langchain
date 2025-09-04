# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os 

# os.environ['HF_HOME'] = 'D:/huggingface_cache'

# llm = HuggingFacePipeline.from_model_id(
#     model_id = "zephyr-7b-beta",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# ) 

# model = ChatHuggingFace(llm= llm)

# result= model.invoke("What is the capital of France")

# print(result.content)

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline # pyright: ignore[reportMissingImports]
import os 

# Set local cache directory (so models don’t redownload every time)
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

# Use full model repo ID
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Create HuggingFace pipeline wrapped for LangChain
llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

# Wrap with Chat interface
model = ChatHuggingFace(llm=llm)

# Run a query
result = model.invoke("What is the capital of France?")

print(result.content)
