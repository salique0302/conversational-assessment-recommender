import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.agent import agent_service
from app.schemas import Message

def test_vague_query_clarification():
    history = [
        Message(role="user", content="I need an assessment")
    ]
    result = agent_service.process_chat(history)
    
    assert len(result["recommendations"]) == 0
    assert result["end_of_conversation"] is False
    # Depending on what the LLM predicts (or the mock), it should ask for missing context
    assert "Could you tell me more about" in result["reply"] or "clarify" in result["reply"].lower()

def test_sufficient_context_recommends():
    # The specific test case requested
    history = [
        Message(role="user", content="I need an assessment"),
        Message(role="assistant", content="Could you tell me more about the role level and specific skills you are testing for?"),
        Message(role="user", content="Hiring a Java developer, mid-level, works with stakeholders")
    ]
    result = agent_service.process_chat(history)
    
    assert len(result["recommendations"]) > 0
    assert result["end_of_conversation"] is False
    assert "Could you tell me more about" not in result["reply"]

def test_over_asking_prevention():
    # Scenario where role and skills are present, but seniority is missing. Score = 2.
    # It should recommend, not ask.
    history = [
        Message(role="user", content="I need a test for a software engineer who knows python and machine learning.")
    ]
    result = agent_service.process_chat(history)
    
    assert len(result["recommendations"]) > 0
    
if __name__ == "__main__":
    pytest.main([__file__])
