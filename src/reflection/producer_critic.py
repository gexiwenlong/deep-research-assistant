import json
import difflib
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.prompts.prompts import CRITIC_SYSTEM_PROMPT, PRODUCER_REVISION_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ProducerCriticLoop:
    def __init__(self, producer_model="gpt-3.5-turbo", critic_model="gpt-3.5-turbo", max_iterations=3, score_threshold=8.0):
        self.producer_model = producer_model
        self.critic_model = critic_model
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

    def _get_critique(self, user_question: str, domain: str, draft: str) -> dict:
        """Get structured critique from the critic."""
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Question: {user_question}\nDomain: {domain}\n\nDraft to Evaluate:\n---\n{draft}\n---"}
        ]
        
        response = client.chat.completions.create(
            model=self.critic_model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("[Critic Error] Failed to parse critic's JSON. Proceeding with caution.")
            return {
                "scores": {"factual_grounding": 5, "completeness": 5, "internal_consistency": 5, "tone_appropriateness": 5, "unsupported_claims": 5},
                "aggregate_score": 5.0,
                "general_critique": "Could not generate meaningful critique.",
                "revision_instructions": "No specific instructions."
            }

    def _get_revision(self, user_question: str, current_draft: str, critique: dict) -> str:
        """Generate a revised draft from the producer."""
        critique_str = critique.get('general_critique', 'No critique available.')
        instructions_str = critique.get('revision_instructions', 'No instructions.')
        
        prompt = PRODUCER_REVISION_PROMPT.format(
            user_question=user_question,
            current_draft=current_draft,
            critique=critique_str,
            instructions=instructions_str
        )
        
        response = client.chat.completions.create(
            model=self.producer_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content

    def reflect(self, user_question: str, domain: str, initial_draft: str):
        print(f"\n{'='*60}\n[REFLECTION] Starting Producer-Critic Loop")
        
        current_draft = initial_draft
        all_scores = []
        plateau_count = 0
        previous_score = -1

        for i in range(1, self.max_iterations + 1):
            print(f"\n[Iteration {i}]")

            print("  [Critic] Evaluating draft...")
            feedback = self._get_critique(user_question, domain, current_draft)
            
            raw_scores = feedback.get('scores', {})
            aggregate_score = sum(raw_scores.values()) / len(raw_scores) if raw_scores else 0
            all_scores.append(aggregate_score)
            print(f"  Scores: {raw_scores}")
            print(f"  Aggregate Score: {aggregate_score:.1f} / 10.0")

            if aggregate_score >= self.score_threshold:
                print(f"  [SUCCESS] Score threshold ({self.score_threshold}) met! Stopping reflection.")
                break

            if abs(aggregate_score - previous_score) < 0.1:
                plateau_count += 1
                print(f"  [WARNING] Score plateau detected (count: {plateau_count}).")
                if plateau_count >= 2:
                    print("  [STOP] Score has plateaued. Stopping reflection.")
                    break
            else:
                plateau_count = 0
            previous_score = aggregate_score

            print("  [Producer] Revising draft...")
            old_draft = current_draft
            current_draft = self._get_revision(user_question, current_draft, feedback)

            print("\n  [Diff] Draft Changes:")
            diff = difflib.unified_diff(
                old_draft.splitlines(keepends=True),
                current_draft.splitlines(keepends=True),
                fromfile='iteration_{}'.format(i-1),
                tofile='iteration_{}'.format(i),
                lineterm=''
            )
            print(''.join(list(diff)[:20]) + "...")

        print(f"\n[REFLECTION COMPLETE] Final Scores across iterations: {all_scores}")
        return current_draft, all_scores
