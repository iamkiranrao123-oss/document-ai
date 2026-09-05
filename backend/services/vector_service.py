import uuid
import chromadb

from backend.config.settings import CHROMA_DB_PATH


client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(chunks, embeddings, filename):
    """
    Store document chunks, embeddings, and metadata in ChromaDB.
    """

    collection.delete(
        where={
            "filename": filename
        }
    )

    ids = []
    metadatas = []

    for chunk in chunks:

        chunk_id = str(uuid.uuid4())

        ids.append(chunk_id)

        metadatas.append({
            "filename": filename,
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


def document_exists(filename):
    """
    Check whether a document exists in ChromaDB.
    """

    results = collection.get(
        where={
            "filename": filename
        },
        include=[]
    )

    return len(results["ids"]) > 0


def search_documents(
    query_embedding,
    n_results=3,
    filename=None
):
    """
    Search ChromaDB for the most relevant document chunks.

    If filename is provided, search only within that document.
    """

    total_documents = collection.count()

    if total_documents == 0:

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    where_filter = None

    if filename:

        where_filter = {
            "filename": filename
        }

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