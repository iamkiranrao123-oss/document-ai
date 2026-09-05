from backend.services.pdf_service import extract_pages
from backend.services.chunk_service import chunk_pages
from backend.services.embedding_service import create_embeddings
from backend.services.vector_service import add_documents, search_documents
from backend.services.rag_service import generate_answer


# PDF location
pdf_path = r"C:\Users\Dell\Desktop\test_document.pdf"


# 1. Extract text from each PDF page
pages = extract_pages(pdf_path)

print("\nPDF pages extracted.")
print("Number of pages:", len(pages))


# 2. Split pages into chunks while preserving page numbers
chunks = chunk_pages(pages)

print("Number of chunks:", len(chunks))


# 3. Extract text from each chunk for embedding
chunk_texts = [chunk["text"] for chunk in chunks]


# 4. Create embeddings
embeddings = create_embeddings(chunk_texts)

print("Embeddings created.")


# 5. Store chunks, embeddings, and metadata in ChromaDB
add_documents(
    chunks,
    embeddings,
    "test_document.pdf"
)

print("Documents stored in ChromaDB.")


# 6. Ask a question
question = "What does the system use to store vectors?"


# 7. Create an embedding for the question
question_embedding = create_embeddings([question])[0]

print("Question embedding created.")


# 8. Search ChromaDB
results = search_documents(
    question_embedding,
    n_results=3
)


# 9. Retrieve matching documents and metadata
retrieved_documents = results["documents"][0]
retrieved_metadata = results["metadatas"][0]


print("\nRetrieved context:")


for i, document in enumerate(retrieved_documents):

    metadata = retrieved_metadata[i]

    print(f"\n--- Result {i + 1} ---")
    print("Source:", metadata["filename"])
    print("Chunk:", metadata["chunk_id"])
    print("Page:", metadata["page_number"])
    print("Text:")
    print(document)


# 10. Combine retrieved chunks
context = "\n\n".join(retrieved_documents)


# 11. Generate answer using Groq
answer = generate_answer(
    question,
    context
)


# 12. Display final answer
print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)