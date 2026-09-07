import uuid

import chromadb

from backend.config.settings import CHROMA_DB_PATH


client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(
    chunks,
    embeddings,
    filename,
    username
):
    """
    Store document chunks, embeddings, and metadata
    in ChromaDB for a specific user.
    """
    collection.delete(
        where={
            "$and": [
                {"filename": filename},
                {"username": username}
            ]
        }
    )

    ids = []
    metadatas = []

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())

        ids.append(chunk_id)

        metadatas.append({
            "filename": filename,
            "username": username,
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"]
        })

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def document_exists(
    filename,
    username
):
    """
    Check whether a document belongs to
    the authenticated user.
    """
    results = collection.get(
        where={
            "$and": [
                {"filename": filename},
                {"username": username}
            ]
        },
        include=[]
    )

    return len(results["ids"]) > 0


def delete_document(
    filename,
    username
):
    """
    Delete all ChromaDB chunks belonging to
    a specific document and authenticated user.
    """
    collection.delete(
        where={
            "$and": [
                {"filename": filename},
                {"username": username}
            ]
        }
    )


def search_documents(
    query_embedding,
    n_results=3,
    filename=None,
    username=None
):
    """
    Search for semantically similar document chunks.

    Results can optionally be restricted to a specific
    filename and authenticated username.
    """
    total_documents = collection.count()

    if total_documents == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    filters = []

    if filename:
        filters.append({
            "filename": filename
        })

    if username:
        filters.append({
            "username": username
        })

    if len(filters) == 1:
        where_filter = filters[0]

    elif len(filters) > 1:
        where_filter = {
            "$and": filters
        }

    else:
        where_filter = None

    if where_filter:
        matching_documents = collection.get(
            where=where_filter,
            include=[]
        )

        matching_count = len(
            matching_documents["ids"]
        )

        if matching_count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        n_results = min(
            n_results,
            matching_count
        )

    else:
        n_results = min(
            n_results,
            total_documents
        )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results


def filter_relevant_results(
    results,
    relevance_threshold
):
    """
    Keep only retrieved chunks whose distance
    is within the configured relevance threshold.

    Lower distance means greater similarity.
    """
    if not results:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    documents = results.get(
        "documents",
        [[]]
    )

    metadatas = results.get(
        "metadatas",
        [[]]
    )

    distances = results.get(
        "distances",
        [[]]
    )

    if not documents or not distances:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for index, distance in enumerate(
        distances[0]
    ):
        if distance <= relevance_threshold:

            filtered_documents.append(
                documents[0][index]
            )

            if metadatas and metadatas[0]:
                filtered_metadatas.append(
                    metadatas[0][index]
                )

            filtered_distances.append(
                distance
            )

    return {
        "documents": [filtered_documents],
        "metadatas": [filtered_metadatas],
        "distances": [filtered_distances]
    }