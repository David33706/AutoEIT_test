"""
LLM-based EIT scoring using OpenAI GPT-4o-mini.
Sends each stimulus-response pair with the rubric to get a 0-4 score.
"""

import json
import time
from openai import OpenAI

RUBRIC_PROMPT = """You are scoring a Spanish Elicited Imitation Task (EIT). A learner heard a Spanish sentence and tried to repeat it. Score their response on a 0-4 scale based on MEANING preservation:

4 = Exact repetition. Form and meaning both correct without exception.
3 = Complete meaning preserved. Grammar changes that don't affect meaning are okay. Omitting 'muy' (very) is acceptable. Substituting 'y'/'pero' is acceptable. Self-corrections, hesitations, and false starts are not penalized — score the best final response.
2 = More than half of idea units preserved. Meaning is close but inexact, incomplete, or ambiguous. String is meaningful.
1 = Only about half of idea units represented. Lots of important information missing. Or string doesn't constitute a self-standing sentence.
0 = Silence, garbled/unintelligible, or minimal repetition (only 1-2 words).

IMPORTANT:
- Missing accents (peliculas vs películas) do NOT affect the score.
- False starts marked with '-' should be ignored. Score the best final attempt.
- [gibberish], [pause], xxx indicate unintelligible or silent portions.
- When in doubt between scores 2 and 3, score 2.
- Score ONLY meaning preservation, not pronunciation or spelling.

Respond with ONLY valid JSON: {"score": N, "reason": "brief explanation"}
Do not include any other text, markdown, or code blocks."""


def score_with_llm(target, response, client):
    """Score a single EIT item using GPT-4o-mini."""
    if not response or response.strip() == "":
        return {"score": 0, "reason": "No response / silence"}

    prompt = f"""Target sentence: {target}
Learner's response: {response}

Score this response:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RUBRIC_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100,
        )

        result_text = completion.choices[0].message.content.strip()

        # Handle markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            result_text = result_text.rsplit("```", 1)[0]

        result = json.loads(result_text)
        return {"score": int(result["score"]), "reason": result.get("reason", "")}

    except Exception as e:
        print(f"    LLM error: {e}")
        return {"score": -1, "reason": f"Error: {e}"}