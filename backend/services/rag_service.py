from groq import Groq

from backend.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL
)


client = Groq(
    api_key=GROQ_API_KEY
)


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved document context.
    """

    prompt = f"""
You are a helpful document analysis assistant.

Answer the user's question using ONLY the provided document context.

Important rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "I could not find the answer in the document."
4. When making a factual claim, cite the page containing
   the supporting information using this format:

   [Page X]

5. Use the page number provided in the source marker.
6. Do not create page numbers that are not present in the context.
7. Keep the answer concise and directly answer the question.

Document context:

{context}

User question:

{question}

Answer:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content