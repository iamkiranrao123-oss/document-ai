from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.embedding_service import create_embeddings
from backend.services.vector_service import (
    search_documents,
    document_exists
)
from backend.services.rag_service import generate_answer
from backend.services.logging_service import get_logger


router = APIRouter()

logger = get_logger("query")


RELEVANCE_THRESHOLD = 1.5


class QueryRequest(BaseModel):
    question: str
    filename: str


@router.post("/query")
async def query_document(request: QueryRequest):

    question = request.question
    filename = request.filename

    logger.info(
        f"Query received | filename={filename}"
    )

    if not question.strip():

        logger.warning(
            "Query rejected | empty question"
        )

        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": []
        }

    try:

        if not document_exists(filename):

            logger.warning(
                f"Document not found | filename={filename}"
            )

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        logger.info(
            f"Document found | filename={filename}"
        )

        question_embedding = create_embeddings(
            [question]
        )[0]

        logger.info(
            "Query embedding created"
        )

        results = search_documents(
            question_embedding,
            n_results=5,
            filename=filename
        )

        retrieved_documents = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        logger.info(
            f"Retrieval complete | "
            f"filename={filename} | "
            f"chunks_retrieved={len(retrieved_documents)}"
        )

        for index, distance in enumerate(distances):

            logger.info(
                f"Retrieved chunk | "
                f"rank={index + 1} | "
                f"distance={distance}"
            )

        if not retrieved_documents:

            logger.warning(
                f"No chunks retrieved | filename={filename}"
            )

            return {
                "question": question,
                "answer": "I could not find the answer in the document.",
                "sources": []
            }

        # Keep only chunks that individually pass
        # the relevance threshold.
        relevant_chunks = []

        for document, metadata, distance in zip(
            retrieved_documents,
            retrieved_metadata,
            distances
        ):

            if distance <= RELEVANCE_THRESHOLD:

                relevant_chunks.append({
                    "document": document,
                    "metadata": metadata,
                    "distance": distance
                })

                logger.info(
                    f"Chunk accepted | "
                    f"chunk_id={metadata['chunk_id']} | "
                    f"page={metadata['page_number']} | "
                    f"distance={distance}"
                )

            else:

                logger.info(
                    f"Chunk rejected | "
                    f"chunk_id={metadata['chunk_id']} | "
                    f"page={metadata['page_number']} | "
                    f"distance={distance}"
                )

        if not relevant_chunks:

            logger.warning(
                f"All retrieved chunks failed relevance threshold | "
                f"filename={filename}"
            )

            return {
                "question": question,
                "answer": "I could not find the answer in the document.",
                "sources": []
            }

        # Send at most 3 relevant chunks to the LLM.
        relevant_chunks = relevant_chunks[:3]

        logger.info(
            f"Relevant chunks selected | "
            f"count={len(relevant_chunks)}"
        )

        # Build source-aware context.
        context_parts = []

        for chunk in relevant_chunks:

            metadata = chunk["metadata"]

            source_header = (
                f"[Source: {metadata['filename']} | "
                f"Page: {metadata['page_number']} | "
                f"Chunk: {metadata['chunk_id']}]"
            )

            context_parts.append(
                f"{source_header}\n"
                f"{chunk['document']}"
            )

        context = "\n\n".join(
            context_parts
        )

        # Generate the final answer.
        answer = generate_answer(
            question,
            context
        )

        logger.info(
            f"LLM answer generated | filename={filename}"
        )

        # Build source information using only the
        # chunks actually sent to the LLM.
        sources = []

        for chunk in relevant_chunks:

            metadata = chunk["metadata"]

            sources.append({
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "chunk_id": metadata["chunk_id"]
            })

        logger.info(
            f"Query completed | "
            f"filename={filename} | "
            f"sources={len(sources)}"
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            f"Unexpected query error | "
            f"filename={filename} | "
            f"error={error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Document query failed."
        )