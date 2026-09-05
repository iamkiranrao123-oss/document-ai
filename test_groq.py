from backend.services.rag_service import generate_answer


question = "What is machine learning?"

context = """
Machine learning is a branch of artificial intelligence.
It allows computers to learn patterns from data and make predictions
without being explicitly programmed for every task.
"""

answer = generate_answer(question, context)

print("\nAnswer:")
print(answer)