from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Schema with Pydantic
class Review(BaseModel):
    
    key_themes: List[str] = Field(..., description="Key themes discussed in the review")
    summary: str = Field(..., description="Concise summary of the review")
    sentiment: str = Field(..., description="Sentiment of the review (Positive, Negative, Neutral)")
    pros: Optional[List[str]] = Field(None, description="Pros mentioned in the review")
    cons: Optional[List[str]] = Field(None, description="Cons mentioned in the review")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say , it's an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-making and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.
""")

print("\n=== Structured Output ===")
print(result)
print("\nSummary:", result.summary)
print("Sentiment:", result.sentiment)
