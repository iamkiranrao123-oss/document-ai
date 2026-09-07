from backend.services.vector_service import (
    filter_relevant_results
)


def test_filter_relevant_results_keeps_relevant_chunks():
    results = {
        "documents": [[
            "relevant one",
            "relevant two",
            "irrelevant"
        ]],
        "metadatas": [[
            {"page_number": 1},
            {"page_number": 2},
            {"page_number": 3}
        ]],
        "distances": [[
            0.2,
            0.7,
            1.8
        ]]
    }

    filtered = filter_relevant_results(
        results,
        relevance_threshold=1.0
    )

    assert filtered["documents"] == [[
        "relevant one",
        "relevant two"
    ]]

    assert filtered["distances"] == [[
        0.2,
        0.7
    ]]


def test_filter_relevant_results_removes_irrelevant_chunks():
    results = {
        "documents": [[
            "irrelevant one",
            "irrelevant two"
        ]],
        "metadatas": [[
            {"page_number": 1},
            {"page_number": 2}
        ]],
        "distances": [[
            1.8,
            2.0
        ]]
    }

    filtered = filter_relevant_results(
        results,
        relevance_threshold=1.0
    )

    assert filtered["documents"] == [[]]
    assert filtered["metadatas"] == [[]]
    assert filtered["distances"] == [[]]


def test_filter_relevant_results_preserves_metadata():
    results = {
        "documents": [[
            "chunk one",
            "chunk two"
        ]],
        "metadatas": [[
            {
                "filename": "test.pdf",
                "page_number": 4,
                "chunk_id": 7
            },
            {
                "filename": "test.pdf",
                "page_number": 8,
                "chunk_id": 12
            }
        ]],
        "distances": [[
            0.3,
            0.6
        ]]
    }

    filtered = filter_relevant_results(
        results,
        relevance_threshold=1.0
    )

    assert filtered["metadatas"] == [[
        {
            "filename": "test.pdf",
            "page_number": 4,
            "chunk_id": 7
        },
        {
            "filename": "test.pdf",
            "page_number": 8,
            "chunk_id": 12
        }
    ]]


def test_filter_relevant_results_preserves_ranking_order():
    results = {
        "documents": [[
            "best chunk",
            "second best",
            "third best"
        ]],
        "metadatas": [[
            {"page_number": 1},
            {"page_number": 2},
            {"page_number": 3}
        ]],
        "distances": [[
            0.1,
            0.4,
            0.9
        ]]
    }

    filtered = filter_relevant_results(
        results,
        relevance_threshold=1.0
    )

    assert filtered["documents"][0] == [
        "best chunk",
        "second best",
        "third best"
    ]

    assert filtered["distances"][0] == [
        0.1,
        0.4,
        0.9
    ]


def test_filter_relevant_results_handles_empty_results():
    results = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }

    filtered = filter_relevant_results(
        results,
        relevance_threshold=1.0
    )

    assert filtered == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }