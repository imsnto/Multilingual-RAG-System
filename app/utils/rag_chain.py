from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List, Dict

from app.config import Settings

class RAGChainBuilder:
    """
    Builds and orchestrates the RAG chain.
    """
    def __init__(self, vector_store: PineconeVectorStore, config: Settings):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0.7,
                              google_api_key=config.GEMINI_API_KEY)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        self.prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI assistant. Use the following context to answer the question. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer. follow the instructions strictly.
        1. Answer the question in Bengali.
        2. If the question is not in Bengali, translate it to Bengali and then answer.
        3. If the question is in Bengali, answer it in Bengali.
        4. Provide answer not directly from the context. Instead, use the context to infer the answer.
        5. If the context is not relevant to the question, say that you don't know.
        6. Do not use any external information.
        7. Short and precise answer.
        8. Donot add extra words or info. Just give answer directly from the context.


        Examples:
        ১. অপরিচিতা গল্পটি কার লেখা?
        রবীন্দ্রনাথ ঠাকুর 

        ২. অনুপমের মামা অনুপমের চেয়ে কত বছরের বড়?
        অনুপমের মামা অনুপমের চেয়ে প্রায় ছয় বছরের বড় ছিলেন।

        ৩. অনুপমের মামা বিয়ের জন্য গহনা পরীক্ষা করতে সাথে কাকে নিয়ে গিয়েছিলেন?
        অনুপমের মামা বিয়ের জন্য গহনা পরীক্ষা করতে সাথে সেকরাকে নিয়ে গিয়েছিলেন।

        Context: {context}"""),
        ("human", "{question}"),
    ]
)

    def format_docs(self, docs: List[Document]) -> str:
        """
        Formats the retrieved documents into a single string for the prompt.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_rag_chain(self):
        """
        Constructs the RAG chain using LangChain Expression Language (LCEL).
        """
        rag_chain = (
            {"context": self.retriever | RunnableLambda(self.format_docs), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain
    
    def get_retrieved_documents(self, query: str) -> List[Document]:
        """
        Retrieves documents for a given query and prints them for inspection.
        """
        docs = self.retriever.invoke(query)
        print(f"\n=== Retrieved {len(docs)} Documents ===\n")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:\n{doc.page_content}\n")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"Metadata: {doc.metadata}\n")
        return docs

if __name__ == "__main__":
    from app.config import Settings
    from app.utils.vector_store import PineconeManager
    from app.utils.data_loader import DocumentProcessor

    config = Settings()
    pinecone_manager = PineconeManager(config)
    try:
        vector_store_instance = pinecone_manager.get_vector_store()
        rag_builder = RAGChainBuilder(vector_store=vector_store_instance, config=config)
        rag_chain = rag_builder.build_rag_chain()

        query = """Tell me about Anupam's character"""
        print(f"Query: {query}")
        response = rag_chain.invoke(query)
        print(f"Response: {response}")

        # x = rag_builder.get_retrieved_documents(query)
    except Exception as e:
        print(f"Could not run RAG chain test. Ensure Pinecone index is populated and API keys are correct: {e}")