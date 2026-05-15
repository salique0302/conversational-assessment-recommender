import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from app.schemas import Message
from app.services.retriever import retriever_service
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ExtractedIntent(BaseModel):
    is_vague: bool = Field(description="True if the user's request lacks specific skills, job roles, or assessment needs.")
    missing_info: List[str] = Field(description="List of specific missing details if the request is vague (e.g. 'role type', 'required skills'). Empty if not vague.")
    search_query: Optional[str] = Field(description="A derived search query for the catalog if the request is specific.")
    filters: Optional[dict] = Field(description="Any extracted metadata filters like 'category'.")

class AgentService:
    def __init__(self):
        # We handle cases where API key is not present by using a fallback mock logic
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def analyze_conversation(self, history: List[Message]) -> ExtractedIntent:
        if not self.client:
            return self._mock_analyze(history)
            
        messages = [
            {"role": "system", "content": "You are a helpful HR assessment assistant. Analyze the user's request. If it's too vague, indicate what information is missing. If it's specific, generate a search query to find the right SHL assessments."}
        ]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
            
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=ExtractedIntent
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Error calling OpenAI, falling back to mock: {e}")
            return self._mock_analyze(history)

    def _mock_analyze(self, history: List[Message]) -> ExtractedIntent:
        """Fallback rule-based logic if no OpenAI API key is set."""
        text = " ".join([m.content for m in history if m.role == "user"]).lower()
        if len(text.split()) < 5 or ("developer" not in text and "sales" not in text and "manager" not in text and "engineer" not in text):
            return ExtractedIntent(
                is_vague=True,
                missing_info=["specific role you are hiring for", "key skills required"],
                search_query=None,
                filters=None
            )
        else:
            return ExtractedIntent(
                is_vague=False,
                missing_info=[],
                search_query=text,
                filters=None
            )

    def process_chat(self, history: List[Message]) -> dict:
        intent = self.analyze_conversation(history)
        
        # Decision logic: Ask vs Recommend
        if intent.is_vague:
            missing = ", ".join(intent.missing_info)
            reply = f"Could you provide more details? I would need to know the {missing} to recommend the right assessments."
            return {
                "reply": reply,
                "recommendations": [],
                "end_of_conversation": False
            }
        else:
            # Generate recommendations
            query = intent.search_query or history[-1].content
            results = retriever_service.semantic_search(query, top_k=3, filters=intent.filters)
            
            # Format reply
            if not results:
                reply = "I couldn't find any specific SHL assessments matching your exact needs, but perhaps you could try rephrasing your requirements?"
            else:
                names = [r['name'] for r in results]
                reply = f"Based on your requirements, I found {len(results)} recommended SHL assessments: {', '.join(names)}. Would you like to know more about any of them?"
                
            return {
                "reply": reply,
                "recommendations": results,
                "end_of_conversation": False
            }

agent_service = AgentService()
