import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from app.schemas import Message, AssessmentRecommendation
from app.services.retriever import retriever_service
from app.services.catalog import catalog_service
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ExtractedIntent(BaseModel):
    action: Literal["clarify", "recommend", "refine", "compare", "complete"] = Field(
        description="The primary action the user is trying to take. 'clarify' if their request is too vague. 'recommend' if they are asking for new recommendations. 'refine' if they are modifying previous requirements. 'compare' if they are asking for the differences between specific assessments. 'complete' if they indicate they are done or satisfied."
    )
    missing_info: List[str] = Field(description="Specific missing details if action is 'clarify' (e.g. 'role type', 'required skills').")
    search_query: Optional[str] = Field(description="A derived search query for the catalog if action is 'recommend' or 'refine'. Incorporate constraints from the full history.")
    filters: Optional[dict] = Field(description="Extracted metadata filters (e.g. category) for the search.")
    comparison_targets: List[str] = Field(description="List of assessment names the user wants to compare, if action is 'compare'.")
    reply_draft: str = Field(description="A draft natural language response handling the request. For 'clarify', ask for the missing info. For 'complete', say goodbye.")

class AgentService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def analyze_conversation(self, history: List[Message]) -> ExtractedIntent:
        if not self.client:
            return self._mock_analyze(history)
            
        messages = [
            {"role": "system", "content": "You are an expert HR assessment advisor. Analyze the user's conversation to determine their intent. If they are vague, choose 'clarify'. If they state new needs, choose 'recommend'. If they change their mind or add constraints, choose 'refine' and update the search_query. If they ask to compare tests, choose 'compare' and list the test names. If they are finished, choose 'complete'."}
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
            print(f"Error calling OpenAI: {e}")
            return self._mock_analyze(history)

    def _mock_analyze(self, history: List[Message]) -> ExtractedIntent:
        text = " ".join([m.content for m in history if m.role == "user"]).lower()
        if "compare" in text:
            return ExtractedIntent(action="compare", missing_info=[], search_query=None, filters=None, comparison_targets=["Verify Numerical Reasoning", "Verify Verbal Reasoning"], reply_draft="")
        elif "thanks" in text or "done" in text:
            return ExtractedIntent(action="complete", missing_info=[], search_query=None, filters=None, comparison_targets=[], reply_draft="You're welcome! Let me know if you need anything else.")
        elif len(text.split()) < 5:
            return ExtractedIntent(action="clarify", missing_info=["role level", "specific skills"], search_query=None, filters=None, comparison_targets=[], reply_draft="Could you tell me more about the role level and specific skills you are testing for?")
        else:
            return ExtractedIntent(action="recommend", missing_info=[], search_query=text, filters=None, comparison_targets=[], reply_draft="")

    def process_chat(self, history: List[Message]) -> dict:
        intent = self.analyze_conversation(history)
        
        reply = ""
        recommendations = []
        end_of_conversation = False
        
        if intent.action == "clarify":
            reply = intent.reply_draft or f"Could you provide more details? I specifically need to know about: {', '.join(intent.missing_info)}."
        
        elif intent.action in ["recommend", "refine"]:
            query = intent.search_query or history[-1].content
            results = retriever_service.semantic_search(query, top_k=3, filters=intent.filters)
            recommendations = results
            
            if not results:
                reply = "I couldn't find any specific SHL assessments matching your exact needs. Could we try adjusting the criteria?"
            else:
                names = [r['name'] for r in results]
                if intent.action == "refine":
                    reply = f"I've updated the recommendations based on your new constraints. Here are {len(results)} assessments that fit better: {', '.join(names)}."
                else:
                    reply = f"Based on your requirements, I found {len(results)} recommended SHL assessments: {', '.join(names)}. Would you like to compare any of them?"
                    
        elif intent.action == "compare":
            if not intent.comparison_targets:
                reply = "Which assessments would you like to compare?"
            else:
                compare_details = []
                for target in intent.comparison_targets:
                    items = retriever_service.semantic_search(target, top_k=1)
                    if items:
                        item = items[0]
                        compare_details.append(f"- **{item['name']}** ({item['category']}): {item['description']} Time limit: {item.get('metadata', {}).get('time_limit', 'N/A')}.")
                
                if compare_details:
                    reply = "Here is a comparison of the requested assessments:\n\n" + "\n".join(compare_details)
                else:
                    reply = "I couldn't find those specific assessments in the catalog to compare."
                    
        elif intent.action == "complete":
            reply = intent.reply_draft or "Thank you for using the SHL Assessment Recommender! Have a great day."
            end_of_conversation = True
            
        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation
        }

agent_service = AgentService()
