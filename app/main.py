from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, AssessmentRecommendation

app = FastAPI(title="SHL Recommender API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Dummy response for Phase 1
    return ChatResponse(
        reply="Hello! I can help you find the right SHL assessment. What kind of role are you hiring for?",
        recommendations=[],
        end_of_conversation=False
    )
