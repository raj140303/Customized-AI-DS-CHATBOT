import os
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_rag_chain(api_key: str):
    # Load model first (used in both scenarios)
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")

    # Use a lightweight embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Define vectorstore path
    vectorstore_path = "vectorstore_db"

    if os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        print("✅ Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    else:
        print("⚙️ Creating vectorstore from scratch...")
        # Load and split documents
        loader = TextLoader("knowledge/ai_ds_knowledge.txt", encoding='utf-8')
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Create and save vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vectorstore_path)
        print("💾 Vectorstore saved at:", vectorstore_path)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt template
    prompt_template = """
    You are an expert assistant in Data Science and AI.
    Answer the following question using the provided context from the knowledge base.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
