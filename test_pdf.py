from backend.services.pdf_service import extract_text
from backend.services.chunk_service import chunk_text
from backend.services.embedding_service import create_embeddings
from backend.services.vector_service import add_documents, search_documents


pdf_path = r"C:\Users\Dell\Desktop\test_document.pdf"

# 1. Extract text
text = extract_text(pdf_path)

# 2. Split text into chunks
chunks = chunk_text(text)

# 3. Create embeddings for the chunks
embeddings = create_embeddings(chunks)

# 4. Store chunks and embeddings
add_documents(chunks, embeddings)

# 5. Create an embedding for the user's question
question = "What does the system use to store vectors?"
question_embedding = create_embeddings([question])[0]

# 6. Search ChromaDB
results = search_documents(question_embedding, n_results=3)

# 7. Display retrieved chunks
print("\nRetrieved chunks:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Result {i + 1} ---")
    print(document)