from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
)

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


logger = get_logger(
    "upload"
)


UPLOAD_DIR = "uploads"


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


def get_user_upload_directory(
    username: str
):
    """
    Create a filesystem-safe directory for
    the authenticated user.

    The username itself is not stored in the
    filesystem path. A SHA-256 hash is used.
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


def sanitize_filename(
    filename: str
):
    """
    Convert an uploaded filename into a safe
    filesystem filename.
    """

    filename = filename.strip()

    if not filename:
        return None

    # Remove Windows and Unix path components.
    filename = re.split(
        r"[\\/]",
        filename
    )[-1]

    # Remove leading/trailing whitespace again.
    filename = filename.strip()

    if not filename:
        return None

    # Only allow PDF files.
    if not filename.lower().endswith(".pdf"):
        return None

    # Reject special path-like names.
    if filename in {".", ".."}:
        return None

    # Replace characters that are unsafe in filenames.
    filename = re.sub(
        r'[^A-Za-z0-9._-]',
        "_",
        filename
    )

    if not filename:
        return None

    return filename


@router.post(
    "/upload",
    summary="Upload a PDF document",
    description=(
        "Uploads a PDF, extracts its text, creates "
        "chunks and embeddings, and stores the "
        "document in the vector database."
    ),
)
async def upload_document(
    file: UploadFile = File(...),
    current_username: str = Depends(
        get_current_username
    )
):
    logger.info(
        "Upload started | username=%s | filename=%s",
        current_username,
        file.filename
    )

    # -------------------------------------------------
    # 1. Validate MIME type
    # -------------------------------------------------

    if file.content_type != "application/pdf":

        logger.warning(
            "Invalid file type | username=%s | "
            "filename=%s | content_type=%s",
            current_username,
            file.filename,
            file.content_type
        )

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # -------------------------------------------------
    # 2. Validate filename
    # -------------------------------------------------

    if not file.filename:

        logger.warning(
            "Upload rejected | username=%s | "
            "reason=filename missing",
            current_username
        )

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    safe_filename = sanitize_filename(
        file.filename
    )

    if not safe_filename:

        logger.warning(
            "Upload rejected | username=%s | "
            "filename=%s | reason=invalid filename",
            current_username,
            file.filename
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid filename. "
                "A PDF filename is required."
            )
        )

    # -------------------------------------------------
    # 3. Create user-specific upload directory
    # -------------------------------------------------

    user_upload_directory = (
        get_user_upload_directory(
            current_username
        )
    )

    file_path = os.path.join(
        user_upload_directory,
        safe_filename
    )

    # -------------------------------------------------
    # 4. Stream file to disk with size protection
    # -------------------------------------------------

    total_size = 0

    try:

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

                    logger.warning(
                        "Upload rejected | username=%s | "
                        "filename=%s | size=%s | "
                        "reason=file too large",
                        current_username,
                        safe_filename,
                        total_size
                    )

                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "PDF file size must not "
                            "exceed 10 MB."
                        )
                    )

                buffer.write(
                    chunk
                )

        logger.info(
            "PDF saved | username=%s | "
            "filename=%s | size=%s",
            current_username,
            safe_filename,
            total_size
        )

        # -------------------------------------------------
        # 5. Extract PDF text
        # -------------------------------------------------

        pages = extract_pages(
            file_path
        )

        logger.info(
            "Text extraction complete | "
            "username=%s | filename=%s | pages=%s",
            current_username,
            safe_filename,
            len(pages)
        )

        if not pages:

            logger.warning(
                "No extractable text | "
                "username=%s | filename=%s",
                current_username,
                safe_filename
            )

            if os.path.exists(file_path):
                os.remove(file_path)

            raise HTTPException(
                status_code=400,
                detail=(
                    "The PDF contains no "
                    "extractable text."
                )
            )

        # -------------------------------------------------
        # 6. Chunk the document
        # -------------------------------------------------

        chunks = chunk_pages(
            pages
        )

        logger.info(
            "Chunking complete | "
            "username=%s | filename=%s | chunks=%s",
            current_username,
            safe_filename,
            len(chunks)
        )

        if not chunks:

            logger.warning(
                "No chunks created | "
                "username=%s | filename=%s",
                current_username,
                safe_filename
            )

            if os.path.exists(file_path):
                os.remove(file_path)

            raise HTTPException(
                status_code=400,
                detail=(
                    "No text chunks could be created "
                    "from the PDF."
                )
            )

        # -------------------------------------------------
        # 7. Create embeddings
        # -------------------------------------------------

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = create_embeddings(
            chunk_texts
        )

        logger.info(
            "Embeddings created | "
            "username=%s | filename=%s | embeddings=%s",
            current_username,
            safe_filename,
            len(embeddings)
        )

        # -------------------------------------------------
        # 8. Store document in ChromaDB
        # -------------------------------------------------

        add_documents(
            chunks,
            embeddings,
            safe_filename,
            current_username
        )

        logger.info(
            "Document stored in vector database | "
            "username=%s | filename=%s",
            current_username,
            safe_filename
        )

        # -------------------------------------------------
        # 9. Successful response
        # -------------------------------------------------

        logger.info(
            "Upload completed successfully | "
            "username=%s | filename=%s",
            current_username,
            safe_filename
        )

        return {
            "message": (
                "Document uploaded successfully"
            ),
            "filename": safe_filename,
            "pages": len(pages),
            "chunks": len(chunks)
        }

    except HTTPException:
        # Preserve intentional HTTP errors.
        if os.path.exists(file_path):
            os.remove(file_path)

        raise

    except Exception as error:

        logger.exception(
            "Unexpected upload error | "
            "username=%s | filename=%s | error=%s",
            current_username,
            safe_filename,
            error
        )

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        )

    finally:
        await file.close()