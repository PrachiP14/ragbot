import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def _init_(self, pdf_path, api_key):
        self.pdf_path = pdf_path
        os.environ["GOOGLE_API_KEY"] = api_key
        self.setup_system()

    def setup_system(self):
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        data = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(data)

        # Setup embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(documents=self.texts, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Setup LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

        # Setup RAG chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def get_answer(self, question):
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]

def run_chatbot(pdf_path, api_key):
    """
    Run a simple chatbot interface for the RAG system
    
    Args:
        pdf_path (str): Path to the PDF file
        api_key (str): Google API key
    """
    print("Initializing RAG system...")
    rag = RAGSystem(pdf_path, api_key)
    
    print("\n" + "="*50)
    print("PDF Question Answering Chatbot")
    print("="*50)
    print("Type 'exit' to quit the chatbot")
    print("="*50 + "\n")
    
    while True:
        question = input("\nYou: ")
        if question.lower() == 'exit':
            print("\nGoodbye!")
            break
            
        print("\nThinking...")
        try:
            answer = rag.get_answer(question)
            print(f"\nBot: {answer}")
        except Exception as e:
            print(f"\nError: {str(e)}")

if _name_ == "_main_":
    # Set your PDF path and API key here
    PDF_PATH = "Collage.pdf"  # Change this to your PDF file path
    API_KEY = "AIzaSyCuSfm9NX-utxP7fnaINGmO8ibpPZOmT9o"  # Add your Google API key here
    
    run_chatbot(PDF_PATH, API_KEY)