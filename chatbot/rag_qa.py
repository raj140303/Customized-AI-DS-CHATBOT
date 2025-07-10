from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_rag_chain(api_key: str):
    # Load knowledge base
    loader = TextLoader("knowledge/ai_ds_knowledge.txt", encoding='utf-8')
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings with lightweight model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define prompt
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

    # Load model
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
