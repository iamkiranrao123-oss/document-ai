import logging

from groq import Groq

from backend.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL
)
from backend.services.logging_service import get_logger


_client = None

logger = get_logger(
    "document_ai.rag"
)


def get_client():
    """
    Create and return the Groq client lazily.

    The client is created only when an LLM
    request is actually needed.
    """

    global _client

    if _client is None:

        if not GROQ_API_KEY:

            logger.error(
                "GROQ_API_KEY is not configured."
            )

            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        _client = Groq(
            api_key=GROQ_API_KEY
        )

    return _client


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved
    document context.
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

    client = get_client()

    try:

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

    except Exception as error:

        logger.exception(
            "LLM request failed | error=%s",
            error
        )

        raise RuntimeError(
            "The language model could not generate an answer."
        ) from error