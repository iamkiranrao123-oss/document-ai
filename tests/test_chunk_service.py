from backend.services.chunk_service import (
    split_into_sentences,
    split_long_sentence,
    chunk_pages,
)


def test_split_into_sentences():
    text = "This is sentence one. This is sentence two! Is this sentence three?"

    sentences = split_into_sentences(text)

    assert sentences == [
        "This is sentence one.",
        "This is sentence two!",
        "Is this sentence three?",
    ]


def test_split_long_sentence():
    sentence = " ".join(["word"] * 300)

    pieces = split_long_sentence(
        sentence,
        chunk_size=100
    )

    assert len(pieces) > 1

    for piece in pieces:
        assert len(piece) <= 100


def test_chunk_pages_preserves_page_number():
    pages = [
        {
            "page_number": 1,
            "text": (
                "This is the first sentence. "
                "This is the second sentence. "
                "This is the third sentence."
            ),
        }
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) >= 1

    for chunk in chunks:
        assert chunk["page_number"] == 1


def test_chunk_pages_assigns_unique_chunk_ids():
    pages = [
        {
            "page_number": 1,
            "text": (
                "This is sentence one. "
                "This is sentence two. "
                "This is sentence three."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "This is another sentence. "
                "This is another sentence."
            ),
        },
    ]

    chunks = chunk_pages(pages)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_pages_ignores_empty_pages():
    pages = [
        {
            "page_number": 1,
            "text": "",
        },
        {
            "page_number": 2,
            "text": "This page contains useful information.",
        },
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 2


def test_chunk_pages_preserves_text():
    pages = [
        {
            "page_number": 1,
            "text": (
                "Document processing is important. "
                "Chunking improves retrieval quality."
            ),
        }
    ]

    chunks = chunk_pages(pages)

    combined_text = " ".join(
        chunk["text"]
        for chunk in chunks
    )

    assert "Document processing is important." in combined_text
    assert "Chunking improves retrieval quality." in combined_text