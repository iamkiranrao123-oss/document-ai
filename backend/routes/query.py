from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_username

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
async def query_document(
    request: QueryRequest,
    current_username: str = Depends(
        get_current_username
    )
):

    question = request.question
    filename = request.filename

    logger.info(
        f"Query received | "
        f"username={current_username} | "
        f"filename={filename}"
    )

    if not question.strip():

        logger.warning(
            f"Query rejected | "
            f"username={current_username} | "
            f"empty question"
        )

        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": []
        }

    try:

        if not document_exists(
            filename,
            current_username
        ):

            logger.warning(
                f"Document not found | "
                f"username={current_username} | "
                f"filename={filename}"
            )

            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        logger.info(
            f"Document found | "
            f"username={current_username} | "
            f"filename={filename}"
        )

        question_embedding = create_embeddings(
            [question]
        )[0]

        logger.info(
            f"Query embedding created | "
            f"username={current_username}"
        )

        results = search_documents(
            question_embedding,
            n_results=5,
            filename=filename,
            username=current_username
        )

        retrieved_documents = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]
        distances = results.get(
            "distances",
            [[]]
        )[0]

        logger.info(
            f"Retrieval complete | "
            f"username={current_username} | "
            f"filename={filename} | "
            f"chunks_retrieved={len(retrieved_documents)}"
        )

        for index, distance in enumerate(distances):

            logger.info(
                f"Retrieved chunk | "
                f"username={current_username} | "
                f"rank={index + 1} | "
                f"distance={distance}"
            )

        if not retrieved_documents:

            logger.warning(
                f"No chunks retrieved | "
                f"username={current_username} | "
                f"filename={filename}"
            )

            return {
                "question": question,
                "answer": "I could not find the answer in the document.",
                "sources": []
            }

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
                    f"username={current_username} | "
                    f"chunk_id={metadata['chunk_id']} | "
                    f"page={metadata['page_number']} | "
                    f"distance={distance}"
                )

            else:

                logger.info(
                    f"Chunk rejected | "
                    f"username={current_username} | "
                    f"chunk_id={metadata['chunk_id']} | "
                    f"page={metadata['page_number']} | "
                    f"distance={distance}"
                )

        if not relevant_chunks:

            logger.warning(
                f"All retrieved chunks failed relevance threshold | "
                f"username={current_username} | "
                f"filename={filename}"
            )

            return {
                "question": question,
                "answer": "I could not find the answer in the document.",
                "sources": []
            }

        relevant_chunks = relevant_chunks[:3]

        logger.info(
            f"Relevant chunks selected | "
            f"username={current_username} | "
            f"count={len(relevant_chunks)}"
        )

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

        answer = generate_answer(
            question,
            context
        )

        logger.info(
            f"LLM answer generated | "
            f"username={current_username} | "
            f"filename={filename}"
        )

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
            f"username={current_username} | "
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
            f"username={current_username} | "
            f"filename={filename} | "
            f"error={error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Document query failed."
        )