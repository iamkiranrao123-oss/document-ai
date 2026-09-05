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

    # Remove the user's previous version of this document.
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
    Check whether a document exists for a specific user.
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


def search_documents(
    query_embedding,
    n_results=3,
    filename=None,
    username=None
):
    """
    Search ChromaDB for the most relevant document chunks.

    If filename is provided, search only within that document.

    If username is provided, search only documents
    belonging to that user.
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