import hashlib
import os
import re

from fastapi import APIRouter, Depends, HTTPException

from backend.auth.dependencies import get_current_username
from backend.services.vector_service import (
    collection,
    document_exists,
    delete_document
)


router = APIRouter()


UPLOAD_DIR = "uploads"


def get_user_upload_directory(
    username: str
):
    """
    Return the filesystem directory belonging
    to the authenticated user.
    """

    user_hash = hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()

    return os.path.join(
        UPLOAD_DIR,
        user_hash
    )


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


@router.delete("/documents/{filename}")
def remove_document(
    filename: str,
    current_username: str = Depends(
        get_current_username
    )
):
    """
    Delete a document belonging to the
    authenticated user.

    Both the ChromaDB vectors and the physical
    uploaded PDF are removed.
    """

    safe_filename = re.split(
        r"[\\/]",
        filename
    )[-1]

    if not safe_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    if not document_exists(
        safe_filename,
        current_username
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    # Delete document chunks from ChromaDB.
    delete_document(
        safe_filename,
        current_username
    )

    # Delete the physical PDF.
    user_upload_directory = (
        get_user_upload_directory(
            current_username
        )
    )

    file_path = os.path.join(
        user_upload_directory,
        safe_filename
    )

    if os.path.exists(file_path):

        try:

            os.remove(file_path)

        except OSError as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Document vectors were deleted, "
                    "but the physical PDF could not "
                    "be removed."
                )
            )

    return {
        "message": "Document deleted successfully.",
        "filename": safe_filename
    }