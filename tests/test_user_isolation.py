import uuid

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import documents


client = TestClient(app)


def create_test_user():
    username = f"user_{uuid.uuid4().hex}"
    password = "testpassword123"

    response = client.post(
        "/register",
        json={
            "username": username,
            "password": password
        }
    )

    assert response.status_code == 200

    return username, password


def login_user(
    username,
    password
):
    response = client.post(
        "/login",
        json={
            "username": username,
            "password": password
        }
    )

    assert response.status_code == 200

    return response.json()["access_token"]


def get_headers(
    username,
    password
):
    token = login_user(
        username,
        password
    )

    return {
        "Authorization": f"Bearer {token}"
    }


def test_user_can_list_own_documents(
    monkeypatch
):
    user_a, password_a = create_test_user()

    def mock_collection_get(
        where,
        include
    ):
        assert where == {
            "username": user_a
        }

        return {
            "metadatas": [
                {
                    "filename": "user_a.pdf",
                    "username": user_a
                }
            ]
        }

    monkeypatch.setattr(
        documents.collection,
        "get",
        mock_collection_get
    )

    headers = get_headers(
        user_a,
        password_a
    )

    response = client.get(
        "/documents",
        headers=headers
    )

    assert response.status_code == 200

    assert response.json() == {
        "documents": [
            "user_a.pdf"
        ]
    }


def test_user_cannot_list_another_users_documents(
    monkeypatch
):
    user_a, password_a = create_test_user()
    user_b, password_b = create_test_user()

    def mock_collection_get(
        where,
        include
    ):
        # The important security property:
        # the authenticated user's username is used
        # as the ChromaDB filter.
        assert where == {
            "username": user_a
        }

        # ChromaDB returns only documents belonging
        # to the authenticated user.
        return {
            "metadatas": []
        }

    monkeypatch.setattr(
        documents.collection,
        "get",
        mock_collection_get
    )

    headers = get_headers(
        user_a,
        password_a
    )

    response = client.get(
        "/documents",
        headers=headers
    )

    assert response.status_code == 200

    assert response.json() == {
        "documents": []
    }

    assert user_a != user_b


def test_user_cannot_query_another_users_document(
    monkeypatch
):
    user_a, password_a = create_test_user()
    user_b, password_b = create_test_user()

    def mock_document_exists(
        filename,
        username
    ):
        assert username == user_a

        return False

    monkeypatch.setattr(
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    headers = get_headers(
        user_a,
        password_a
    )

    response = client.post(
        "/query",
        json={
            "question": "What is in the document?",
            "filename": "user_b.pdf"
        },
        headers=headers
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }

    assert user_a != user_b


def test_user_cannot_delete_another_users_document(
    monkeypatch
):
    user_a, password_a = create_test_user()
    user_b, password_b = create_test_user()

    def mock_document_exists(
        filename,
        username
    ):
        assert username == user_a

        return False

    monkeypatch.setattr(
        "backend.routes.documents.document_exists",
        mock_document_exists
    )

    headers = get_headers(
        user_a,
        password_a
    )

    response = client.delete(
        "/documents/user_b.pdf",
        headers=headers
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }

    assert user_a != user_b


def test_same_filename_isolated_between_users(
    monkeypatch
):
    user_a, password_a = create_test_user()
    user_b, password_b = create_test_user()

    calls = []

    def mock_add_documents(
        chunks,
        embeddings,
        filename,
        username
    ):
        calls.append(
            {
                "filename": filename,
                "username": username
            }
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

    monkeypatch.setattr(
        "backend.routes.upload.add_documents",
        mock_add_documents
    )

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

    headers_a = get_headers(
        user_a,
        password_a
    )

    headers_b = get_headers(
        user_b,
        password_b
    )

    response_a = client.post(
        "/upload",
        files={
            "file": (
                "same.pdf",
                b"fake pdf content",
                "application/pdf"
            )
        },
        headers=headers_a
    )

    response_b = client.post(
        "/upload",
        files={
            "file": (
                "same.pdf",
                b"fake pdf content",
                "application/pdf"
            )
        },
        headers=headers_b
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    assert len(calls) == 2

    assert calls[0]["filename"] == "same.pdf"
    assert calls[1]["filename"] == "same.pdf"

    assert calls[0]["username"] == user_a
    assert calls[1]["username"] == user_b

    assert calls[0]["username"] != calls[1]["username"]