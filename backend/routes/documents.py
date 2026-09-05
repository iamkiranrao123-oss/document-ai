from fastapi import APIRouter

from backend.services.vector_service import collection


router = APIRouter()


@router.get("/documents")
def list_documents():
    """
    Return the unique PDF filenames stored in ChromaDB.
    """

    results = collection.get(
        include=["metadatas"]
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    filenames = sorted({
        metadata["filename"]
        for metadata in metadatas
        if metadata.get("filename")
    })

    return {
        "documents": filenames
    }