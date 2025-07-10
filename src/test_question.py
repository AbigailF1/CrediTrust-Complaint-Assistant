from rag.generator import generate_answer

question = "What are the most common complaints about delayed payments?"
answer, sources = generate_answer(question)

print("🔹 Answer:\n", answer)
print("\n🔸 Top Sources:")
for i, doc in enumerate(sources):
    print(f"\nSource {i+1}:\n{doc.page_content[:300]}")
