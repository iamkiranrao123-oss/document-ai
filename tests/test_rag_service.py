from backend.services import rag_service


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [
            FakeChoice(content)
        ]


class FakeCompletions:
    def __init__(self):
        self.last_call = None

    def create(
        self,
        model,
        messages,
        temperature
    ):
        self.last_call = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        return FakeResponse(
            "The answer is supported by the document. [Page 2]"
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_generate_answer_uses_document_context(
    monkeypatch
):
    fake_client = FakeClient()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: fake_client
    )

    question = "What is RAG?"

    context = (
        "[Source: test.pdf | Page: 2 | Chunk: 1]\n"
        "RAG stands for Retrieval-Augmented Generation."
    )

    answer = rag_service.generate_answer(
        question,
        context
    )

    assert answer == (
        "The answer is supported by the document. [Page 2]"
    )

    prompt = (
        fake_client
        .chat
        .completions
        .last_call["messages"][0]["content"]
    )

    assert context in prompt
    assert question in prompt


def test_generate_answer_uses_configured_model(
    monkeypatch
):
    fake_client = FakeClient()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: fake_client
    )

    rag_service.generate_answer(
        "What is RAG?",
        "RAG is Retrieval-Augmented Generation."
    )

    call = (
        fake_client
        .chat
        .completions
        .last_call
    )

    assert call["model"] == rag_service.GROQ_MODEL


def test_generate_answer_uses_low_temperature(
    monkeypatch
):
    fake_client = FakeClient()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: fake_client
    )

    rag_service.generate_answer(
        "What is RAG?",
        "RAG is Retrieval-Augmented Generation."
    )

    call = (
        fake_client
        .chat
        .completions
        .last_call
    )

    assert call["temperature"] == 0.1


def test_generate_answer_prompt_contains_grounding_rules(
    monkeypatch
):
    fake_client = FakeClient()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: fake_client
    )

    rag_service.generate_answer(
        "What is RAG?",
        "RAG is Retrieval-Augmented Generation."
    )

    prompt = (
        fake_client
        .chat
        .completions
        .last_call["messages"][0]["content"]
    )

    assert "ONLY the provided document context" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "Do not invent information" in prompt
    assert (
        "I could not find the answer in the document."
        in prompt
    )


def test_generate_answer_prompt_requires_page_citations(
    monkeypatch
):
    fake_client = FakeClient()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: fake_client
    )

    rag_service.generate_answer(
        "What is RAG?",
        "[Source: test.pdf | Page: 5 | Chunk: 2]\n"
        "RAG retrieves relevant information."
    )

    prompt = (
        fake_client
        .chat
        .completions
        .last_call["messages"][0]["content"]
    )

    assert "[Page X]" in prompt
    assert "Do not create page numbers" in prompt


def test_generate_answer_handles_llm_failure(
    monkeypatch
):
    class FailingCompletions:
        def create(
            self,
            model,
            messages,
            temperature
        ):
            raise Exception(
                "Simulated Groq failure"
            )

    class FailingChat:
        def __init__(self):
            self.completions = FailingCompletions()

    class FailingClient:
        def __init__(self):
            self.chat = FailingChat()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: FailingClient()
    )

    try:
        rag_service.generate_answer(
            "What is RAG?",
            "RAG retrieves relevant information."
        )

        assert False, (
            "Expected RuntimeError "
            "was not raised."
        )

    except RuntimeError as error:
        assert str(error) == (
            "The language model could not "
            "generate an answer."
        )


def test_generate_answer_logs_llm_failure(
    monkeypatch,
    caplog
):
    class FailingCompletions:
        def create(
            self,
            model,
            messages,
            temperature
        ):
            raise Exception(
                "Simulated Groq failure"
            )

    class FailingChat:
        def __init__(self):
            self.completions = FailingCompletions()

    class FailingClient:
        def __init__(self):
            self.chat = FailingChat()

    monkeypatch.setattr(
        rag_service,
        "get_client",
        lambda: FailingClient()
    )

    with caplog.at_level("ERROR"):
        try:
            rag_service.generate_answer(
                "What is RAG?",
                "RAG retrieves relevant information."
            )
        except RuntimeError:
            pass

    assert any(
        "LLM request failed" in record.message
        for record in caplog.records
    )