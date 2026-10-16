# support_assistant/app/schemas.py
# This file defines the Pydantic models for the request and response schemas used in the support assistant API.

from pydantic import BaseModel, Field, ValidationError

# Pydantic models provide data validation and serialization for the API endpoints. 
# The AskRequest model represents the user's query, while the SupportResponse model represents the assistant's answer along with the sources and confidence score. 
# The confidence score is validated to be between 0.0 and 1.0.

class AskRequest(BaseModel):
    query: str = Field(..., description="The user's query to the support assistant.")

class SupportResponse(BaseModel):
    answer: str = Field(..., description="The support assistant's answer to the user's query.")
    sources: list[str] = Field(default_factory=list, description="The source documents from which the answer was derived.")
    confidence: float = Field(ge=0.0, le=1.0, description="The confidence score of the answer, between 0.0 and 1.0.")


if __name__ == "__main__":

    # Test the SupportResponse model with valid data

    response = SupportResponse(
        answer="This is the support assistant's answer.",
        sources=["doc1", "doc2"],
        confidence=0.95
    )
    print(response)

    # Test the SupportResponse model with invalid data

    try:
        invalid_response = SupportResponse(
        answer="This is the support assistant's answer.",
        sources=["doc1", "doc2"],
        confidence=1.5
    )
    except ValidationError as e:
        print("Error:", e)