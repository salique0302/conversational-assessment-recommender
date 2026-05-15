from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, AssessmentRecommendation
from app.services.agent import agent_service

app = FastAPI(title="SHL Recommender API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_service.process_chat(request.conversation_history)
    
    return ChatResponse(
        reply=result["reply"],
        recommendations=result["recommendations"],
        end_of_conversation=result["end_of_conversation"]
    )

