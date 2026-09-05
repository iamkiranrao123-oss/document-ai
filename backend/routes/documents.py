from fastapi import APIRouter, Depends

from backend.auth.dependencies import get_current_username
from backend.services.vector_service import collection


router = APIRouter()


@router.get("/documents")
def list_documents(
    current_username: str = Depends(
        get_current_username
    )
):
    """
    Return the unique PDF filenames belonging
    to the authenticated user.
    """

    results = collection.get(
        where={
            "username": current_username
        },
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