from app.loader import load_cv_text
from app.embedder import CVVectorStore
from app.rag import RAGPipeline
from app.chat import openai_llm

# Load CV
text = load_cv_text()

# Build vector store
store = CVVectorStore()
store.chunk_text(text)
store.create_index()

# Build pipeline
rag = RAGPipeline(store, openai_llm)

# Ask questions
print("Chatbot ready!")
while True:
    q = input("Me: ")
    ans = rag.ask(q)
    print("Bot: ", ans)