from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from pathlib import Path

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


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    logger.info(
        f"Upload started | filename={file.filename}"
    )

    if file.content_type != "application/pdf":

        logger.warning(
            f"Invalid file type | filename={file.filename} "
            f"| content_type={file.content_type}"
        )

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    if not file.filename:

        logger.warning(
            "Upload rejected | filename missing"
        )

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    safe_filename = Path(
        file.filename
    ).name

    if not safe_filename:

        logger.warning(
            "Upload rejected | invalid filename"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    try:

        file_path = os.path.join(
            UPLOAD_DIR,
            safe_filename
        )

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
            f"filename={safe_filename} | "
            f"size={total_size}"
        )

        pages = extract_pages(
            file_path
        )

        logger.info(
            f"Text extraction complete | "
            f"filename={safe_filename} | "
            f"pages={len(pages)}"
        )

        if not pages:

            logger.warning(
                f"No extractable text | "
                f"filename={safe_filename}"
            )

            raise HTTPException(
                status_code=400,
                detail="The PDF contains no extractable text."
            )

        chunks = chunk_pages(
            pages
        )

        logger.info(
            f"Chunking complete | "
            f"filename={safe_filename} | "
            f"chunks={len(chunks)}"
        )

        if not chunks:

            logger.warning(
                f"No chunks created | "
                f"filename={safe_filename}"
            )

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
            f"filename={safe_filename} | "
            f"embeddings={len(embeddings)}"
        )

        add_documents(
            chunks,
            embeddings,
            safe_filename
        )

        logger.info(
            f"Document stored in vector database | "
            f"filename={safe_filename}"
        )

        logger.info(
            f"Upload completed successfully | "
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
            f"filename={safe_filename} | "
            f"error={error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        )