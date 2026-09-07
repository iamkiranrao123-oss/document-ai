import io

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def get_auth_headers():
    """
    Register a temporary test user and return
    the JWT Authorization header.
    """

    import uuid

    username = f"upload_test_{uuid.uuid4().hex}"
    password = "testpassword123"

    register_response = client.post(
        "/register",
        json={
            "username": username,
            "password": password
        }
    )

    assert register_response.status_code == 200

    login_response = client.post(
        "/login",
        json={
            "username": username,
            "password": password
        }
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}"
    }


def test_upload_rejects_non_pdf_content_type():
    headers = get_auth_headers()

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                io.BytesIO(
                    b"This is not a PDF."
                ),
                "text/plain"
            )
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


def test_upload_rejects_missing_filename():
    headers = get_auth_headers()

    response = client.post(
        "/upload",
        files={
            "file": (
                None,
                io.BytesIO(b""),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf_filename():
    headers = get_auth_headers()

    response = client.post(
        "/upload",
        files={
            "file": (
                "document.txt",
                io.BytesIO(
                    b"fake pdf content"
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Invalid filename. "
            "A PDF filename is required."
        )
    }


def test_upload_sanitizes_path_traversal_filename(
    monkeypatch
):
    headers = get_auth_headers()

    def mock_extract_pages(
        file_path
    ):
        return [
            {
                "page_number": 1,
                "text": "Test document content."
            }
        ]

    def mock_chunk_pages(
        pages
    ):
        return [
            {
                "chunk_id": "chunk-1",
                "page_number": 1,
                "text": "Test document content."
            }
        ]

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
        ]

    def mock_add_documents(
        chunks,
        embeddings,
        filename,
        username
    ):
        pass

    monkeypatch.setattr(
        "backend.routes.upload.extract_pages",
        mock_extract_pages
    )

    monkeypatch.setattr(
        "backend.routes.upload.chunk_pages",
        mock_chunk_pages
    )

    monkeypatch.setattr(
        "backend.routes.upload.create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        "backend.routes.upload.add_documents",
        mock_add_documents
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "../../dangerous.pdf",
                io.BytesIO(
                    b"fake pdf content"
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "dangerous.pdf"


def test_upload_rejects_file_larger_than_limit(
    monkeypatch
):
    headers = get_auth_headers()

    from backend.routes import upload

    monkeypatch.setattr(
        upload,
        "MAX_FILE_SIZE",
        100
    )

    large_content = b"x" * 101

    response = client.post(
        "/upload",
        files={
            "file": (
                "large.pdf",
                io.BytesIO(
                    large_content
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "PDF file size must not "
            "exceed 10 MB."
        )
    }


def test_upload_rejects_pdf_with_no_extractable_text(
    monkeypatch
):
    headers = get_auth_headers()

    def mock_extract_pages(
        file_path
    ):
        return []

    monkeypatch.setattr(
        "backend.routes.upload.extract_pages",
        mock_extract_pages
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.pdf",
                io.BytesIO(
                    b"fake pdf content"
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "The PDF contains no "
            "extractable text."
        )
    }


def test_upload_rejects_when_no_chunks_created(
    monkeypatch
):
    headers = get_auth_headers()

    def mock_extract_pages(
        file_path
    ):
        return [
            {
                "page_number": 1,
                "text": "Some document text."
            }
        ]

    def mock_chunk_pages(
        pages
    ):
        return []

    monkeypatch.setattr(
        "backend.routes.upload.extract_pages",
        mock_extract_pages
    )

    monkeypatch.setattr(
        "backend.routes.upload.chunk_pages",
        mock_chunk_pages
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "nochunks.pdf",
                io.BytesIO(
                    b"fake pdf content"
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "No text chunks could be created "
            "from the PDF."
        )
    }


def test_upload_processing_failure_returns_500(
    monkeypatch
):
    headers = get_auth_headers()

    def mock_extract_pages(
        file_path
    ):
        raise RuntimeError(
            "PDF processing failed."
        )

    monkeypatch.setattr(
        "backend.routes.upload.extract_pages",
        mock_extract_pages
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "failure.pdf",
                io.BytesIO(
                    b"fake pdf content"
                ),
                "application/pdf"
            )
        },
        headers=headers
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Document processing failed."
    }