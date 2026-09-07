import uuid

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def get_auth_headers():
    """
    Register a unique test user, log in,
    and return the Authorization header.
    """

    username = f"testuser_{uuid.uuid4().hex}"
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_rag_query(monkeypatch):
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
            "documents": [
                [
                    "ChromaDB stores document embeddings."
                ]
            ],
            "metadatas": [
                [
                    {
                        "filename": "test.pdf",
                        "username": "testuser",
                        "chunk_id": "chunk-1",
                        "page_number": 4
                    }
                ]
            ],
            "distances": [
                [0.2]
            ]
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
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        "backend.routes.query.create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        "backend.routes.query.search_documents",
        mock_search_documents
    )

    monkeypatch.setattr(
        "backend.routes.query.generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/query",
        json={
            "question": "What stores the embeddings?",
            "filename": "test.pdf"
        },
        headers=headers
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

    assert data["sources"][0] == {
        "filename": "test.pdf",
        "page_number": 4,
        "chunk_id": "chunk-1",
        "distance": 0.2
    }


def test_empty_question():
    headers = get_auth_headers()

    response = client.post(
        "/query",
        json={
            "question": "   ",
            "filename": "test.pdf"
        },
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == ""

    assert data["answer"] == (
        "Please enter a question."
    )

    assert data["sources"] == []


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
            "documents": [
                [
                    "This document discusses "
                    "machine learning."
                ]
            ],
            "metadatas": [
                [
                    {
                        "filename": "test.pdf",
                        "username": "testuser",
                        "chunk_id": "chunk-1",
                        "page_number": 1
                    }
                ]
            ],
            "distances": [
                [2.0]
            ]
        }

    monkeypatch.setattr(
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        "backend.routes.query.create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        "backend.routes.query.search_documents",
        mock_search_documents
    )

    response = client.post(
        "/query",
        json={
            "question": "What is the capital of France?",
            "filename": "test.pdf"
        },
        headers=headers
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
            "documents": [
                [
                    "First relevant chunk.",
                    "Second relevant chunk."
                ]
            ],
            "metadatas": [
                [
                    {
                        "filename": "test.pdf",
                        "username": "testuser",
                        "chunk_id": "chunk-1",
                        "page_number": 2
                    },
                    {
                        "filename": "test.pdf",
                        "username": "testuser",
                        "chunk_id": "chunk-2",
                        "page_number": 5
                    }
                ]
            ],
            "distances": [
                [0.1, 0.3]
            ]
        }

    def mock_generate_answer(
        question,
        context
    ):
        return "Answer from the document."

    monkeypatch.setattr(
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        "backend.routes.query.create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        "backend.routes.query.search_documents",
        mock_search_documents
    )

    monkeypatch.setattr(
        "backend.routes.query.generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/query",
        json={
            "question": "What is discussed?",
            "filename": "test.pdf"
        },
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Answer from the document."
    )

    assert len(data["sources"]) == 2

    assert data["sources"][0] == {
        "filename": "test.pdf",
        "page_number": 2,
        "chunk_id": "chunk-1",
        "distance": 0.1
    }

    assert data["sources"][1] == {
        "filename": "test.pdf",
        "page_number": 5,
        "chunk_id": "chunk-2",
        "distance": 0.3
    }


def test_rag_handles_llm_failure(
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
            "documents": [
                [
                    "This is test document content."
                ]
            ],
            "metadatas": [
                [
                    {
                        "filename": "test.pdf",
                        "username": "testuser",
                        "chunk_id": "chunk-1",
                        "page_number": 1
                    }
                ]
            ],
            "distances": [
                [0.2]
            ]
        }

    def mock_generate_answer(
        question,
        context
    ):
        raise RuntimeError(
            "The language model could not generate an answer."
        )

    monkeypatch.setattr(
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    monkeypatch.setattr(
        "backend.routes.query.create_embeddings",
        mock_create_embeddings
    )

    monkeypatch.setattr(
        "backend.routes.query.search_documents",
        mock_search_documents
    )

    monkeypatch.setattr(
        "backend.routes.query.generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/query",
        json={
            "question": "What is in the document?",
            "filename": "test.pdf"
        },
        headers=headers
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "The language model is temporarily "
            "unavailable. Please try again later."
        )
    }


def test_query_rejects_missing_filename():
    headers = get_auth_headers()

    response = client.post(
        "/query",
        json={
            "question": "What is in the document?",
            "filename": "   "
        },
        headers=headers
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Filename cannot be empty."
    }


def test_query_rejects_unknown_document(
    monkeypatch
):
    headers = get_auth_headers()

    def mock_document_exists(
        filename,
        username
    ):
        return False

    monkeypatch.setattr(
        "backend.routes.query.document_exists",
        mock_document_exists
    )

    response = client.post(
        "/query",
        json={
            "question": "What is in the document?",
            "filename": "missing.pdf"
        },
        headers=headers
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }