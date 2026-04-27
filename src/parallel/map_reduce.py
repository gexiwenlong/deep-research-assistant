import json
import asyncio
import time
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from src.prompts.prompts import DECOMPOSE_PROMPT, SUB_QUESTION_ANSWER_PROMPT, BEST_OF_N_JUDGE_PROMPT, SYNTHESIS_PROMPT, DOMAIN_SYSTEM_PROMPTS

load_dotenv()
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sync_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if 'openai' in dir() else None

class ResearchMapper:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def decompose(self, user_question: str, domain: str) -> list:
        """Break down a research question into 3-5 sub-questions."""
        prompt = DECOMPOSE_PROMPT.format(user_question=user_question, domain=domain)
        response = await async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        try:
            result = json.loads(response.choices[0].message.content)
            sub_questions = result.get("sub_questions", [])
            if isinstance(sub_questions, list) and len(sub_questions) > 0:
                return sub_questions
            else:
                return [user_question]
        except (json.JSONDecodeError, KeyError):
            return [user_question]

class BestOfNJudge:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def select_best(self, sub_question: str, candidates: list) -> str:
        """Select the best candidate answer using an LLM judge."""
        candidates_formatted = "\n\n".join([f"--- Candidate {i} ---\n{c}" for i, c in enumerate(candidates)])
        prompt = BEST_OF_N_JUDGE_PROMPT.format(sub_question=sub_question, candidates_formatted=candidates_formatted)
        
        response = await async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            best_id = result["best_candidate_id"]
            return candidates[best_id]
        except (json.JSONDecodeError, KeyError, IndexError):
            print(f"[Best-of-N Error] Falling back to first candidate. Raw output: {response.choices[0].message.content}")
            return candidates[0] if candidates else "Error: No candidates available."

class DeepResearcher:
    def __init__(self, best_of_n=3, model="gpt-3.5-turbo"):
        self.model = model
        self.best_of_n = best_of_n
        self.mapper = ResearchMapper(model)
        self.judge = BestOfNJudge(model)
        self.domain_personas = DOMAIN_SYSTEM_PROMPTS

    async def _answer_sub_question(self, sub_question: str, persona: str) -> list:
        """Generate N candidate answers for a sub-question."""
        prompt = SUB_QUESTION_ANSWER_PROMPT.format(sub_question=sub_question, persona=persona)
        tasks = [self._call_llm(prompt) for _ in range(self.best_of_n)]
        return await asyncio.gather(*tasks)

    async def _call_llm(self, prompt):
        """Helper to make a single async LLM call."""
        try:
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Parallel Error] Failed to get answer: {e}")
            return "Error: Could not generate response."

    async def _process_single_leaf(self, sub_question: str, persona: str) -> str:
        """The full Best-of-N process for a single sub-question."""
        print(f"  [Parallel] Processing sub-question: {sub_question[:50]}...")
        candidates = await self._answer_sub_question(sub_question, persona)
        valid_candidates = [c for c in candidates if "Error:" not in c]
        if not valid_candidates:
            return "Error: All candidates failed."
        best_answer = await self.judge.select_best(sub_question, valid_candidates)
        return best_answer

    async def research_async(self, user_question: str, domain: str):
        """The main Map-Reduce-Best-of-N pipeline."""
        print(f"\n{'='*60}\n[MAP] Decomposing question...")
        start_time = time.time()
        sub_questions = await self.mapper.decompose(user_question, domain)
        print(f"  Sub-Questions:\n" + "\n".join(f"    - {q}" for q in sub_questions))
        
        persona = self.domain_personas.get(domain, "Be helpful and balanced.")
        print(f"\n[FAN-OUT] Generating {self.best_of_n} candidate answers for each of {len(sub_questions)} sub-questions in parallel...")
        
        parallel_start = time.time()
        tasks = [self._process_single_leaf(q, persona) for q in sub_questions]
        best_answers = await asyncio.gather(*tasks)
        parallel_time = time.time() - parallel_start
        print(f"\n[FAN-IN] All parallel tasks complete in {parallel_time:.2f}s.")
        
        print(f"[REDUCE] Synthesizing final brief...")
        answers_formatted = "\n\n".join([f"**Sub-Q: {q}**\n{a}" for q, a in zip(sub_questions, best_answers)])
        synthesis_prompt = SYNTHESIS_PROMPT.format(user_question=user_question, domain=domain, answers_formatted=answers_formatted)
        
        synthesis_response = await async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3
        )
        
        final_brief = synthesis_response.choices[0].message.content
        total_time = time.time() - start_time
        
        print(f"\n[SUCCESS] Research pipeline completed in {total_time:.2f}s.")
        
        N = len(sub_questions) * self.best_of_n
        estimated_seq_time = N * 1.5
        print(f"  Estimated sequential time: ~{estimated_seq_time:.2f}s (for {N} calls)")
        print(f"  Actual parallel time: {parallel_time:.2f}s")
        print(f"  Estimated speedup: {estimated_seq_time/parallel_time:.2f}x")

        return final_brief
