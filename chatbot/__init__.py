from chatbot.rag_qa import load_rag_chain

class Chatbot:
    """Main chatbot class to handle user interactions via RAG."""
    
    def __init__(self, api_key):
        self.qa_chain = load_rag_chain(api_key)

    def ask(self, question):
        """Ask a question and get RAG-based response."""
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"⚠️ Error: {e}"
