from backend.routes import query as query_module


def test_query_logs_retrieval(
    monkeypatch,
    caplog
):
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
                "ChromaDB stores embeddings."
            ]],
            "metadatas": [[
                {
                    "filename": filename,
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
        return "ChromaDB stores embeddings."

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

    with caplog.at_level("INFO"):
        response = query_module.query_document(
            request=query_module.QueryRequest(
                question="What stores embeddings?",
                filename="test.pdf"
            ),
            current_username="testuser"
        )

    assert response["answer"] == (
        "ChromaDB stores embeddings."
    )

    log_messages = [
        record.message
        for record in caplog.records
    ]

    assert any(
        "Query received" in message
        for message in log_messages
    )

    assert any(
        "Documents retrieved" in message
        for message in log_messages
    )

    assert any(
        "Relevance filtering completed" in message
        for message in log_messages
    )

    assert any(
        "Answer generated" in message
        for message in log_messages
    )


def test_query_logs_no_relevant_information(
    monkeypatch,
    caplog
):
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
                "ChromaDB stores embeddings."
            ]],
            "metadatas": [[
                {
                    "filename": filename,
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

    with caplog.at_level("INFO"):
        response = query_module.query_document(
            request=query_module.QueryRequest(
                question="What is the capital of France?",
                filename="test.pdf"
            ),
            current_username="testuser"
        )

    assert response["answer"] == (
        "I could not find the answer "
        "in the document."
    )

    assert response["sources"] == []

    log_messages = [
        record.message
        for record in caplog.records
    ]

    assert any(
        "No relevant information found" in message
        for message in log_messages
    )