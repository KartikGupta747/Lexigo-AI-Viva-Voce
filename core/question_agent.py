from core.llm import gemini_llm
import random



def generate_viva_questions(retrieved_chunks, num_questions=5):
    seed = random.randint(100000, 999999)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict university viva voce examiner.
RANDOM_SEED: {seed}
You are examining a student on a specific research paper.
Your job is to test DEEP understanding, not memorization.

RULES (VERY IMPORTANT):
- Ask EXACTLY {num_questions} questions.
- Each question must be based ONLY on the provided context.
- Do NOT ask generic questions (e.g., title, authors, abstract).
- Prefer WHY, HOW, and TRADE-OFF questions.
- Each question must test conceptual understanding.
- Questions must be clearly phrased and unambiguous.
- Do NOT provide answers.
- Number the questions from 1 to {num_questions}.

Context from the research paper:
--------------------
{context}
--------------------

Now generate the questions.
"""

    return gemini_llm(prompt)
