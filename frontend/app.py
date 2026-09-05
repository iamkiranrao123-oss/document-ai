import streamlit as st
import requests
import os


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
)

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Document AI",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------
# Session State
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


# -----------------------------------
# Header
# -----------------------------------

st.title("📄 Document AI")

st.markdown(
    """
    **AI-powered document intelligence**

    Upload a PDF, select your document, and ask questions
    using retrieval-augmented generation (RAG).
    """
)


# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.header("📚 Documents")

    st.caption(
        "Upload and select the document you want to analyze."
    )

    # -----------------------------------
    # Upload
    # -----------------------------------

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "Upload Document",
            use_container_width=True
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=120
                    )

                if response.status_code == 200:

                    data = response.json()

                    st.success(
                        "Document uploaded successfully!"
                    )

                    st.write(
                        f"📄 **Pages:** {data['pages']}"
                    )

                    st.write(
                        f"🧩 **Chunks:** {data['chunks']}"
                    )

                    st.rerun()

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "Document processing took too long."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI backend."
                )

    st.divider()

    # -----------------------------------
    # Get Documents
    # -----------------------------------

    try:

        documents_response = requests.get(
            f"{API_URL}/documents",
            timeout=10
        )

        if documents_response.status_code == 200:

            documents = (
                documents_response
                .json()
                .get("documents", [])
            )

        else:

            documents = []

    except requests.exceptions.RequestException:

        documents = []


    # -----------------------------------
    # Document Selector
    # -----------------------------------

    if documents:

        selected_document = st.selectbox(
            "Select document",
            documents
        )

        st.session_state.selected_document = (
            selected_document
        )

        st.success(
            f"Selected: {selected_document}"
        )

    else:

        st.info(
            "No documents uploaded yet."
        )


    st.divider()

    # -----------------------------------
    # Clear Chat
    # -----------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# -----------------------------------
# Main Chat Area
# -----------------------------------

st.header("💬 Ask Questions")


if st.session_state.selected_document:

    st.caption(
        f"Analyzing: **{st.session_state.selected_document}**"
    )

else:

    st.info(
        "Upload and select a document to begin."
    )


# -----------------------------------
# Display Conversation History
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # -----------------------------------
        # Display Sources
        # -----------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                f"📑 Sources ({len(message['sources'])})"
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"""
                        **Source {index}**

                        📄 **Document:** {source['filename']}

                        📖 **Page:** {source['page_number']}

                        🧩 **Chunk:** {source['chunk_id']}
                        """
                    )

                    if index < len(message["sources"]):

                        st.divider()


# -----------------------------------
# Chat Input
# -----------------------------------

question = st.chat_input(
    "Ask a question about your document..."
)


if question:

    # -----------------------------------
    # Check Document
    # -----------------------------------

    if not st.session_state.selected_document:

        st.warning(
            "Please upload and select a document first."
        )

        st.stop()


    # -----------------------------------
    # Display User Question
    # -----------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------
    # Query Backend
    # -----------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your document..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "filename": (
                            st.session_state.selected_document
                        )
                    },
                    timeout=60
                )


                # -----------------------------------
                # Successful Response
                # -----------------------------------

                if response.status_code == 200:

                    data = response.json()

                    answer = data["answer"]

                    st.markdown(answer)


                    # -----------------------------------
                    # Sources
                    # -----------------------------------

                    sources = data.get(
                        "sources",
                        []
                    )

                    if sources:

                        with st.expander(
                            f"📑 Sources ({len(sources)})"
                        ):

                            for index, source in enumerate(
                                sources,
                                start=1
                            ):

                                st.markdown(
                                    f"""
                                    **Source {index}**

                                    📄 **Document:** {source['filename']}

                                    📖 **Page:** {source['page_number']}

                                    🧩 **Chunk:** {source['chunk_id']}
                                    """
                                )

                                if index < len(sources):

                                    st.divider()


                    # -----------------------------------
                    # Save Assistant Message
                    # -----------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )


                # -----------------------------------
                # Document Not Found
                # -----------------------------------

                elif response.status_code == 404:

                    st.error(
                        "The selected document could not be found."
                    )


                # -----------------------------------
                # Other Errors
                # -----------------------------------

                else:

                    st.error(
                        f"Request failed: {response.text}"
                    )


            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the FastAPI backend. "
                    "Make sure Uvicorn is running."
                )


            except requests.exceptions.RequestException as error:

                st.error(
                    f"Request error: {error}"
                )