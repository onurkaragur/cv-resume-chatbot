# Updated imports for modern LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import streamlit as st
from langchain_ibm import WatsonxLLM

# App title
st.title("Ask about me!")

# Load PDF and create vectorstore only once
@st.cache_resource
def load_index():
    loader = PyPDFLoader("about_me.pdf")  # change your file name
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

vectorstore = load_index()

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Watsonx LLM
llm = WatsonxLLM(
    model="ibm/granite-20b",
    apikey="YOUR_API_KEY",
    project_id="YOUR_PROJECT_ID",
)

# RetrievalQA chain (modern)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User input
prompt = st.chat_input("Pass your questions here!")

if prompt:
    # Show user's message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run retrieval QA
    response = qa_chain({"query": prompt})

    answer = response["result"]

    # Show assistant message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
