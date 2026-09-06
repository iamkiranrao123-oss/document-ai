def test_delete_document(client, auth_headers, tmp_path, monkeypatch):
    filename = "delete_test.pdf"
    username = "testuser"

    monkeypatch.setattr(
        "backend.routes.documents.document_exists",
        lambda filename, username: True
    )

    deleted_documents = []

    monkeypatch.setattr(
        "backend.routes.documents.delete_document",
        lambda filename, username: deleted_documents.append(
            (filename, username)
        )
    )

    user_directory = tmp_path / "uploads" / "testuser_hash"
    user_directory.mkdir(parents=True)

    pdf_path = user_directory / filename
    pdf_path.write_bytes(b"fake pdf content")

    monkeypatch.setattr(
        "backend.routes.documents.get_user_upload_directory",
        lambda username: str(user_directory)
    )

    response = client.delete(
        f"/documents/{filename}",
        headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["filename"] == filename
    assert deleted_documents == [(filename, username)]
    assert not pdf_path.exists()


def test_delete_nonexistent_document(client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.documents.document_exists",
        lambda filename, username: False
    )

    response = client.delete(
        "/documents/nonexistent.pdf",
        headers=auth_headers
    )

    assert response.status_code == 404


def test_delete_document_sanitizes_filename(client, auth_headers, monkeypatch):
    deleted_documents = []

    monkeypatch.setattr(
        "backend.routes.documents.document_exists",
        lambda filename, username: True
    )

    monkeypatch.setattr(
        "backend.routes.documents.delete_document",
        lambda filename, username: deleted_documents.append(
            (filename, username)
        )
    )

    response = client.delete(
        "/documents/../delete_test.pdf",
        headers=auth_headers
    )

    assert response.status_code == 200
    assert deleted_documents[0][0] == "delete_test.pdf"