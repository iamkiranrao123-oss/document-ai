from io import BytesIO
import os

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import upload as upload_module
from backend.routes import query as query_module
from backend.routes import documents as documents_module


client = TestClient(app)


TEST_USERNAME = "test_api_user"
TEST_PASSWORD = "test_api_password"


def get_auth_headers():
    """
    Register and log in a test user,
    then return the JWT authorization headers.
    """

    client.post(
        "/register",
        json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD
        }
    )

    response = client.post(
        "/login",
        json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD
        }
    )

    assert response.status_code == 200

    token = response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}"
    }


def create_test_pdf():
    """
    Create a minimal PDF-like byte stream
    for upload tests.
    """

    return BytesIO(
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [] /Count 0 >>\n"
        b"endobj\n"
        b"trailer\n"
        b"<< /Root 1 0 R >>\n"
        b"%%EOF"
    )


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Document AI API is running"
    }


def test_documents_endpoint():

    headers = get_auth_headers()

    response = client.get(
        "/documents",
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data

    assert isinstance(
        data["documents"],
        list
    )


def test_upload_rejects_non_pdf():

    headers = get_auth_headers()

    response = client.post(
        "/upload",
        headers=headers,
        files={
            "file": (
                "test.txt",
                b"This is not a PDF.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Only PDF files are allowed."
    )


def test_upload_rejects_oversized_pdf(
    monkeypatch
):

    headers = get_auth_headers()

    monkeypatch.setattr(
        upload_module,
        "MAX_FILE_SIZE",
        10
    )

    response = client.post(
        "/upload",
        headers=headers,
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * 100,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "PDF file size must not exceed 10 MB."
    )


def test_empty_question():

    headers = get_auth_headers()

    response = client.post(
        "/query",
        headers=headers,
        json={
            "question": "   ",
            "filename": "test.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Please enter a question."
    )

    assert data["sources"] == []


def test_query_nonexistent_document():

    headers = get_auth_headers()

    response = client.post(
        "/query",
        headers=headers,
        json={
            "question": "What is this document about?",
            "filename": "does_not_exist.pdf"
        }
    )

    assert response.status_code == 404

    assert response.json()["detail"] == (
        "Document not found."
    )


def test_rag_query(
    monkeypatch
):

    headers = get_auth_headers()

    def mock_document_exists(
        filename,
        username
    ):
        return True

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
        ]

    def mock_search_documents(
        query_embedding,
        n_results=3,
        filename=None,
        username=None
    ):
        return {
            "documents": [[
                "ChromaDB is used to store "
                "document embeddings."
            ]],
            "metadatas": [[
                {
                    "filename": "test.pdf",
                    "username": username,
                    "chunk_id": "chunk_1",
                    "page_number": 1
                }
            ]],
            "distances": [[
                0.2
            ]]
        }

    def mock_generate_answer(
        question,
        context
    ):
        return (
            "The system uses ChromaDB "
            "to store embeddings."
        )

    monkeypatch.setattr(
        query_module,
        "document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        query_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        query_module,
        "search_documents",
        mock_search_documents
    )

    monkeypatch.setattr(
        query_module,
        "generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/query",
        headers=headers,
        json={
            "question": "What stores the embeddings?",
            "filename": "test.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What stores the embeddings?"
    )

    assert data["answer"] == (
        "The system uses ChromaDB "
        "to store embeddings."
    )

    assert len(data["sources"]) == 1


def test_rag_rejects_unrelated_question(
    monkeypatch
):

    headers = get_auth_headers()

    def mock_document_exists(
        filename,
        username
    ):
        return True

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
        ]

    def mock_search_documents(
        query_embedding,
        n_results=3,
        filename=None,
        username=None
    ):
        return {
            "documents": [[
                "ChromaDB is used to store "
                "document embeddings."
            ]],
            "metadatas": [[
                {
                    "filename": "test.pdf",
                    "username": username,
                    "chunk_id": "chunk_1",
                    "page_number": 1
                }
            ]],
            "distances": [[
                2.0
            ]]
        }

    monkeypatch.setattr(
        query_module,
        "document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        query_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        query_module,
        "search_documents",
        mock_search_documents
    )

    response = client.post(
        "/query",
        headers=headers,
        json={
            "question": "What is the capital of France?",
            "filename": "test.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "I could not find the answer "
        "in the document."
    )

    assert data["sources"] == []


def test_rag_sources_have_correct_metadata(
    monkeypatch
):

    headers = get_auth_headers()

    def mock_document_exists(
        filename,
        username
    ):
        return True

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
        ]

    def mock_search_documents(
        query_embedding,
        n_results=3,
        filename=None,
        username=None
    ):
        return {
            "documents": [[
                "This is information "
                "from the document."
            ]],
            "metadatas": [[
                {
                    "filename": "test.pdf",
                    "username": username,
                    "chunk_id": "chunk_123",
                    "page_number": 3
                }
            ]],
            "distances": [[
                0.1
            ]]
        }

    def mock_generate_answer(
        question,
        context
    ):
        return "This is the answer."

    monkeypatch.setattr(
        query_module,
        "document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        query_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        query_module,
        "search_documents",
        mock_search_documents
    )

    monkeypatch.setattr(
        query_module,
        "generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/query",
        headers=headers,
        json={
            "question": "What does the document say?",
            "filename": "test.pdf"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["sources"]) == 1

    source = data["sources"][0]

    assert source["filename"] == "test.pdf"
    assert source["page_number"] == 3
    assert source["chunk_id"] == "chunk_123"

    assert isinstance(
        source["page_number"],
        int
    )

    assert isinstance(
        source["chunk_id"],
        str
    )


def test_upload_sanitizes_filename(
    monkeypatch,
    tmp_path
):

    headers = get_auth_headers()

    monkeypatch.setattr(
        upload_module,
        "UPLOAD_DIR",
        str(tmp_path)
    )

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
                "chunk_id": "chunk_1",
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
        return None

    monkeypatch.setattr(
        upload_module,
        "extract_pages",
        mock_extract_pages
    )

    monkeypatch.setattr(
        upload_module,
        "chunk_pages",
        mock_chunk_pages
    )

    monkeypatch.setattr(
        upload_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        upload_module,
        "add_documents",
        mock_add_documents
    )

    unsafe_filename = (
        "..\\..\\test_document.pdf"
    )

    response = client.post(
        "/upload",
        headers=headers,
        files={
            "file": (
                unsafe_filename,
                create_test_pdf(),
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == (
        "test_document.pdf"
    )

    assert data["pages"] == 1
    assert data["chunks"] == 1


def test_users_cannot_access_each_others_documents(
    monkeypatch
):
    """
    Verify that one user cannot access another
    user's documents.
    """

    user_a = {
        "username": "isolation_user_a",
        "password": "password_a"
    }

    user_b = {
        "username": "isolation_user_b",
        "password": "password_b"
    }

    response = client.post(
        "/register",
        json=user_a
    )

    assert response.status_code in [200, 409]

    response = client.post(
        "/register",
        json=user_b
    )

    assert response.status_code in [200, 409]

    response = client.post(
        "/login",
        json=user_a
    )

    assert response.status_code == 200

    token_a = response.json()["access_token"]

    headers_a = {
        "Authorization": f"Bearer {token_a}"
    }

    response = client.post(
        "/login",
        json=user_b
    )

    assert response.status_code == 200

    token_b = response.json()["access_token"]

    headers_b = {
        "Authorization": f"Bearer {token_b}"
    }

    def mock_extract_pages(
        file_path
    ):
        return [
            {
                "page_number": 1,
                "text": "This document belongs to User A."
            }
        ]

    def mock_chunk_pages(
        pages
    ):
        return [
            {
                "chunk_id": "isolation_chunk_a",
                "page_number": 1,
                "text": "This document belongs to User A."
            }
        ]

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def mock_add_documents(
        chunks,
        embeddings,
        filename,
        username
    ):
        return None

    monkeypatch.setattr(
        upload_module,
        "extract_pages",
        mock_extract_pages
    )

    monkeypatch.setattr(
        upload_module,
        "chunk_pages",
        mock_chunk_pages
    )

    monkeypatch.setattr(
        upload_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        upload_module,
        "add_documents",
        mock_add_documents
    )

    response = client.post(
        "/upload",
        headers=headers_a,
        files={
            "file": (
                "user_a_document.pdf",
                create_test_pdf(),
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200

    response = client.get(
        "/documents",
        headers=headers_b
    )

    assert response.status_code == 200

    documents = response.json()["documents"]

    assert "user_a_document.pdf" not in documents

    response = client.post(
        "/query",
        headers=headers_b,
        json={
            "question": "What does this document contain?",
            "filename": "user_a_document.pdf"
        }
    )

    assert response.status_code == 404

    assert response.json()["detail"] == (
        "Document not found."
    )


def test_users_can_upload_same_filename_without_overwriting(
    monkeypatch,
    tmp_path
):
    """
    Verify that two different users can upload
    the same filename without overwriting each
    other's physical files.
    """

    user_a = {
        "username": "same_file_user_a",
        "password": "password_a"
    }

    user_b = {
        "username": "same_file_user_b",
        "password": "password_b"
    }

    response = client.post(
        "/register",
        json=user_a
    )

    assert response.status_code in [200, 409]

    response = client.post(
        "/register",
        json=user_b
    )

    assert response.status_code in [200, 409]

    response = client.post(
        "/login",
        json=user_a
    )

    assert response.status_code == 200

    token_a = response.json()["access_token"]

    headers_a = {
        "Authorization": f"Bearer {token_a}"
    }

    response = client.post(
        "/login",
        json=user_b
    )

    assert response.status_code == 200

    token_b = response.json()["access_token"]

    headers_b = {
        "Authorization": f"Bearer {token_b}"
    }

    monkeypatch.setattr(
        upload_module,
        "UPLOAD_DIR",
        str(tmp_path)
    )

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
                "chunk_id": "same_filename_chunk",
                "page_number": 1,
                "text": "Test document content."
            }
        ]

    def mock_create_embeddings(
        texts
    ):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def mock_add_documents(
        chunks,
        embeddings,
        filename,
        username
    ):
        return None

    monkeypatch.setattr(
        upload_module,
        "extract_pages",
        mock_extract_pages
    )

    monkeypatch.setattr(
        upload_module,
        "chunk_pages",
        mock_chunk_pages
    )

    monkeypatch.setattr(
        upload_module,
        "create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        upload_module,
        "add_documents",
        mock_add_documents
    )

    filename = "report.pdf"

    response_a = client.post(
        "/upload",
        headers=headers_a,
        files={
            "file": (
                filename,
                create_test_pdf(),
                "application/pdf"
            )
        }
    )

    assert response_a.status_code == 200

    response_b = client.post(
        "/upload",
        headers=headers_b,
        files={
            "file": (
                filename,
                create_test_pdf(),
                "application/pdf"
            )
        }
    )

    assert response_b.status_code == 200

    user_a_directory = (
        upload_module.get_user_upload_directory(
            user_a["username"]
        )
    )

    user_b_directory = (
        upload_module.get_user_upload_directory(
            user_b["username"]
        )
    )

    user_a_directory_name = os.path.basename(
        user_a_directory
    )

    user_b_directory_name = os.path.basename(
        user_b_directory
    )

    user_a_file = (
        tmp_path
        / user_a_directory_name
        / filename
    )

    user_b_file = (
        tmp_path
        / user_b_directory_name
        / filename
    )

    assert user_a_file.exists()

    assert user_b_file.exists()

    assert user_a_file != user_b_file


def test_delete_document(
    monkeypatch,
    tmp_path
):
    """
    Verify that deleting a document removes both
    its vector records and its physical PDF.
    """

    headers = get_auth_headers()

    filename = "delete_test.pdf"
    username = TEST_USERNAME

    monkeypatch.setattr(
        documents_module,
        "document_exists",
        lambda filename, username: True
    )

    deleted_documents = []

    monkeypatch.setattr(
        documents_module,
        "delete_document",
        lambda filename, username: deleted_documents.append(
            (filename, username)
        )
    )

    user_directory = (
        tmp_path
        / "testuser_hash"
    )

    user_directory.mkdir(
        parents=True
    )

    pdf_path = (
        user_directory
        / filename
    )

    pdf_path.write_bytes(
        b"fake pdf content"
    )

    monkeypatch.setattr(
        documents_module,
        "get_user_upload_directory",
        lambda username: str(user_directory)
    )

    response = client.delete(
        f"/documents/{filename}",
        headers=headers
    )

    assert response.status_code == 200

    assert response.json()["filename"] == filename

    assert deleted_documents == [
        (filename, username)
    ]

    assert not pdf_path.exists()


def test_delete_nonexistent_document(
    monkeypatch
):
    """
    Verify that deleting a document that does not
    exist returns a 404 response.
    """

    headers = get_auth_headers()

    monkeypatch.setattr(
        documents_module,
        "document_exists",
        lambda filename, username: False
    )

    response = client.delete(
        "/documents/nonexistent.pdf",
        headers=headers
    )

    assert response.status_code == 404

    assert response.json()["detail"] == (
        "Document not found."
    )


def test_delete_document_sanitizes_filename(
    monkeypatch
):
    """
    Verify that path components are removed from
    the filename before deletion.
    """

    headers = get_auth_headers()

    deleted_documents = []

    monkeypatch.setattr(
        documents_module,
        "document_exists",
        lambda filename, username: True
    )

    monkeypatch.setattr(
        documents_module,
        "delete_document",
        lambda filename, username: deleted_documents.append(
            (filename, username)
        )
    )

    response = client.delete(
        "/documents/folder\\delete_test.pdf",
        headers=headers
    )

    assert response.status_code == 200

    assert deleted_documents[0][0] == (
        "delete_test.pdf"
    )