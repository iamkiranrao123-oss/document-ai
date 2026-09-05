from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

import hashlib
import os
import re

from backend.auth.dependencies import get_current_username
from backend.config.settings import MAX_FILE_SIZE

from backend.services.pdf_service import extract_pages
from backend.services.chunk_service import chunk_pages
from backend.services.embedding_service import create_embeddings
from backend.services.vector_service import add_documents
from backend.services.logging_service import get_logger


router = APIRouter()

logger = get_logger("upload")


UPLOAD_DIR = "uploads"


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


def get_user_upload_directory(
    username: str
):
    """
    Create a filesystem-safe directory name
    for the authenticated user.

    A SHA-256 hash is used instead of storing
    the username directly in the filesystem path.
    """

    user_hash = hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()

    user_directory = os.path.join(
        UPLOAD_DIR,
        user_hash
    )

    os.makedirs(
        user_directory,
        exist_ok=True
    )

    return user_directory


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_username: str = Depends(
        get_current_username
    )
):

    logger.info(
        f"Upload started | "
        f"username={current_username} | "
        f"filename={file.filename}"
    )

    if file.content_type != "application/pdf":

        logger.warning(
            f"Invalid file type | "
            f"username={current_username} | "
            f"filename={file.filename} | "
            f"content_type={file.content_type}"
        )

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    if not file.filename:

        logger.warning(
            f"Upload rejected | "
            f"username={current_username} | "
            f"filename missing"
        )

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    safe_filename = re.split(
        r"[\\/]",
        file.filename
    )[-1]

    if not safe_filename:

        logger.warning(
            f"Upload rejected | "
            f"username={current_username} | "
            f"invalid filename"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    user_upload_directory = (
        get_user_upload_directory(
            current_username
        )
    )

    file_path = os.path.join(
        user_upload_directory,
        safe_filename
    )

    try:

        total_size = 0

        with open(
            file_path,
            "wb"
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:

                    buffer.close()

                    if os.path.exists(file_path):
                        os.remove(file_path)

                    logger.warning(
                        f"Upload rejected | "
                        f"username={current_username} | "
                        f"filename={safe_filename} | "
                        f"size={total_size} | "
                        f"reason=file too large"
                    )

                    raise HTTPException(
                        status_code=400,
                        detail="PDF file size must not exceed 10 MB."
                    )

                buffer.write(chunk)

        logger.info(
            f"PDF saved | "
            f"username={current_username} | "
            f"filename={safe_filename} | "
            f"size={total_size}"
        )

        pages = extract_pages(
            file_path
        )

        logger.info(
            f"Text extraction complete | "
            f"username={current_username} | "
            f"filename={safe_filename} | "
            f"pages={len(pages)}"
        )

        if not pages:

            logger.warning(
                f"No extractable text | "
                f"username={current_username} | "
                f"filename={safe_filename}"
            )

            if os.path.exists(file_path):
                os.remove(file_path)

            raise HTTPException(
                status_code=400,
                detail="The PDF contains no extractable text."
            )

        chunks = chunk_pages(
            pages
        )

        logger.info(
            f"Chunking complete | "
            f"username={current_username} | "
            f"filename={safe_filename} | "
            f"chunks={len(chunks)}"
        )

        if not chunks:

            logger.warning(
                f"No chunks created | "
                f"username={current_username} | "
                f"filename={safe_filename}"
            )

            if os.path.exists(file_path):
                os.remove(file_path)

            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created from the PDF."
            )

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(
            chunk_texts
        )

        logger.info(
            f"Embeddings created | "
            f"username={current_username} | "
            f"filename={safe_filename} | "
            f"embeddings={len(embeddings)}"
        )

        add_documents(
            chunks,
            embeddings,
            safe_filename,
            current_username
        )

        logger.info(
            f"Document stored in vector database | "
            f"username={current_username} | "
            f"filename={safe_filename}"
        )

        logger.info(
            f"Upload completed successfully | "
            f"username={current_username} | "
            f"filename={safe_filename}"
        )

        return {
            "message": "Document uploaded successfully",
            "filename": safe_filename,
            "pages": len(pages),
            "chunks": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            f"Unexpected upload error | "
            f"username={current_username} | "
            f"filename={safe_filename} | "
            f"error={error}"
        )

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        )