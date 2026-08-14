from typing import List, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(..., example="What is the delivery fee for orders below 149?")

class SupportResponse(BaseModel):
    answer: str = Field(..., description="The generated or templated answer to the user query.")
    sources: List[str] = Field(default_factory=list, description="Document IDs used for answering.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")