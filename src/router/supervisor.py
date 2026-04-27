import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.prompts.prompts import ROUTER_SYSTEM_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResearchSupervisor:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def classify(self, user_question: str) -> dict:
        """Classify a user question into a domain, handling rejection and fallback."""
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content
        
        try:
            result = json.loads(raw_output)
            result.setdefault("classification", "fallback_out_of_scope")
            result.setdefault("reasoning", "JSON parsing failed.")
            result.setdefault("confidence", 0.0)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Routing Error] Failed to parse LLM output: {e}. Raw output: {raw_output}")
            return {
                "classification": "fallback_out_of_scope",
                "reasoning": "An internal error occurred during routing.",
                "confidence": 0.0
            }
