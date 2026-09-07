from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_username
from backend.config.settings import (
    TOP_K,
    MAX_CONTEXT_CHUNKS,
    RELEVANCE_THRESHOLD
)
from backend.services.embedding_service import create_embeddings
from backend.services.rag_service import generate_answer
from backend.services.vector_service import (
    document_exists,
    search_documents,
    filter_relevant_results
)
from backend.services.logging_service import get_logger


router = APIRouter()

logger = get_logger(
    "document_ai.query"
)


class QueryRequest(BaseModel):
    question: str
    filename: str


@router.post("/query")
def query_document(
    request: QueryRequest,
    current_username: str = Depends(
        get_current_username
    )
):
    question = request.question.strip()
    filename = request.filename.strip()

    logger.info(
        "Query received | user=%s | filename=%s",
        current_username,
        filename
    )

    if not question:
        logger.info(
            "Empty question | user=%s | filename=%s",
            current_username,
            filename
        )

        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": []
        }

    if not filename:
        logger.warning(
            "Empty filename | user=%s",
            current_username
        )

        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty."
        )

    if not document_exists(
        filename,
        current_username
    ):
        logger.warning(
            "Document not found | user=%s | filename=%s",
            current_username,
            filename
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        query_embedding = create_embeddings(
            [question]
        )[0]

        logger.info(
            "Query embedding generated | "
            "user=%s | filename=%s",
            current_username,
            filename
        )

    except Exception as error:
        logger.exception(
            "Embedding generation failed | "
            "user=%s | filename=%s | error=%s",
            current_username,
            filename,
            error
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The embedding service is temporarily "
                "unavailable. Please try again later."
            )
        )

    try:
        results = search_documents(
            query_embedding=query_embedding,
            n_results=TOP_K,
            filename=filename,
            username=current_username
        )

        retrieved_documents = len(
            results.get(
                "documents",
                [[]]
            )[0]
        )

        logger.info(
            "Documents retrieved | "
            "user=%s | filename=%s | count=%s",
            current_username,
            filename,
            retrieved_documents
        )

    except Exception as error:
        logger.exception(
            "Vector search failed | "
            "user=%s | filename=%s | error=%s",
            current_username,
            filename,
            error
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The document search service is temporarily "
                "unavailable. Please try again later."
            )
        )

    results = filter_relevant_results(
        results,
        relevance_threshold=RELEVANCE_THRESHOLD
    )

    relevant_documents = len(
        results.get(
            "documents",
            [[]]
        )[0]
    )

    distances = results.get(
        "distances",
        [[]]
    )[0]

    logger.info(
        "Relevance filtering completed | "
        "user=%s | filename=%s | "
        "relevant=%s | distances=%s",
        current_username,
        filename,
        relevant_documents,
        distances
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    if not documents:
        logger.info(
            "No relevant information found | "
            "user=%s | filename=%s",
            current_username,
            filename
        )

        return {
            "question": question,
            "answer": (
                "I could not find the answer "
                "in the document."
            ),
            "sources": []
        }

    selected_count = min(
        len(documents),
        MAX_CONTEXT_CHUNKS
    )

    selected_documents = documents[
        :selected_count
    ]

    selected_metadatas = metadatas[
        :selected_count
    ]

    selected_distances = distances[
        :selected_count
    ]

    logger.info(
        "Context selected | "
        "user=%s | filename=%s | chunks=%s",
        current_username,
        filename,
        selected_count
    )

    context_parts = []

    for document, metadata in zip(
        selected_documents,
        selected_metadatas
    ):
        context_parts.append(
            (
                f"[Source: {metadata['filename']} | "
                f"Page: {metadata['page_number']} | "
                f"Chunk: {metadata['chunk_id']}]\n"
                f"{document}"
            )
        )

    context = "\n\n".join(
        context_parts
    )

    logger.info(
        "Sending context to LLM | "
        "user=%s | filename=%s",
        current_username,
        filename
    )

    try:
        answer = generate_answer(
            question,
            context
        )

    except RuntimeError as error:
        logger.error(
            "LLM service unavailable | "
            "user=%s | filename=%s | error=%s",
            current_username,
            filename,
            error
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The language model is temporarily "
                "unavailable. Please try again later."
            )
        )

    logger.info(
        "Answer generated | "
        "user=%s | filename=%s",
        current_username,
        filename
    )

    sources = []

    for metadata, distance in zip(
        selected_metadatas,
        selected_distances
    ):
        sources.append({
            "filename": metadata["filename"],
            "page_number": metadata["page_number"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }