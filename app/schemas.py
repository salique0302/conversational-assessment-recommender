from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    conversation_history: List[Message] = Field(default_factory=list, alias="messages")

class AssessmentRecommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[AssessmentRecommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
