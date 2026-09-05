from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes.upload import MAX_FILE_SIZE


client = TestClient(app)

TEST_PDF = Path(__file__).parent / "fixtures" / "test_document.pdf"


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Document AI API is running"
    }


def test_documents_endpoint():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_upload_rejects_non_pdf():
    files = {
        "file": (
            "test.txt",
            b"This is not a PDF file.",
            "text/plain"
        )
    }

    response = client.post(
        "/upload",
        files=files
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


def test_upload_rejects_oversized_pdf():

    oversized_content = b"x" * (MAX_FILE_SIZE + 1)

    files = {
        "file": (
            "large.pdf",
            oversized_content,
            "application/pdf"
        )
    }

    response = client.post(
        "/upload",
        files=files
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "PDF file size must not exceed 10 MB."
    }


def test_empty_question():
    response = client.post(
        "/query",
        json={
            "question": "",
            "filename": "test_document.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Please enter a question."
    assert data["sources"] == []


def test_query_nonexistent_document():
    response = client.post(
        "/query",
        json={
            "question": "What is this document about?",
            "filename": "does_not_exist.pdf"
        }
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }


@patch("backend.routes.query.generate_answer")
def test_rag_query(mock_generate_answer):

    mock_generate_answer.return_value = (
        "The system uses a vector database to store the embeddings."
    )

    with open(TEST_PDF, "rb") as pdf_file:

        files = {
            "file": (
                "test_document.pdf",
                pdf_file,
                "application/pdf"
            )
        }

        upload_response = client.post(
            "/upload",
            files=files
        )

    assert upload_response.status_code == 200

    query_response = client.post(
        "/query",
        json={
            "question": "What does the system use to store embeddings?",
            "filename": "test_document.pdf"
        }
    )

    assert query_response.status_code == 200

    data = query_response.json()

    assert data["answer"] == (
        "The system uses a vector database to store the embeddings."
    )

    assert len(data["sources"]) > 0

    mock_generate_answer.assert_called_once()


def test_rag_rejects_unrelated_question():

    response = client.post(
        "/query",
        json={
            "question": "What is the capital of France?",
            "filename": "test_document.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "I could not find the answer in the document."
    )

    assert data["sources"] == []


def test_rag_sources_have_correct_metadata():

    with open(TEST_PDF, "rb") as pdf_file:

        files = {
            "file": (
                "test_document.pdf",
                pdf_file,
                "application/pdf"
            )
        }

        upload_response = client.post(
            "/upload",
            files=files
        )

    assert upload_response.status_code == 200

    response = client.post(
        "/query",
        json={
            "question": "What does the system use to store embeddings?",
            "filename": "test_document.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["sources"]) > 0

    for source in data["sources"]:

        assert source["filename"] == "test_document.pdf"

        assert isinstance(
            source["page_number"],
            int
        )

        assert isinstance(
            source["chunk_id"],
            int
        )

        assert source["page_number"] >= 1

        assert source["chunk_id"] >= 0


def test_upload_sanitizes_filename():

    with open(TEST_PDF, "rb") as pdf_file:

        files = {
            "file": (
                "..\\..\\unsafe_test_document.pdf",
                pdf_file,
                "application/pdf"
            )
        }

        response = client.post(
            "/upload",
            files=files
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "unsafe_test_document.pdf"