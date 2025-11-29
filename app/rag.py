class RAGPipeline:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def ask(self, question):
        retrieved_chunks = self.vector_store.search(question)

        context = "\n".join(retrieved_chunks)

        prompt = f"""
You are an assistant that answers questions strictly based on this CV:
--- CV START ---
{context}
--- CV END ---

Question: {question}
Answer using only the CV. If you cannot find an answer in the CV, say "This information is not in the CV."
"""
        return self.llm(prompt)