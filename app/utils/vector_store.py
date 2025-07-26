from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List

from app.utils.data_loader import DocumentProcessor
from app.config import Settings
import time

class PineconeManager:
    """
    Manages connections and operations with the Pinecone vector database.
    """
    def __init__(self, config: Settings):
        self.config = config
        self.pinecone = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.EMBEDDING_MODEL_NAME, google_api_key=self.config.GEMINI_API_KEY)
        self.index_name = self.config.PINECONE_INDEX_NAME

    def _create_index_if_not_exists(self, dimension: int = 3072, metric: str = "cosine"):
        """
        Creates a Pinecone index if it doesn't already exist.
        """
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating Pinecone index '{self.index_name}'...")
            self.pinecone.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={
                    "environment": "development"
                }
            )
            # Wait for index to be ready
            while not self.pinecone.describe_index(self.index_name).status['ready']:
                print("Waiting for index to be ready...")
                time.sleep(1)
            print(f"Pinecone index '{self.index_name}' created successfully.")
        else:
            print(f"Pinecone index '{self.index_name}' already exists.")

    def get_vector_store(self, dimension: int = 3072) -> PineconeVectorStore:
        """
        Connects to the Pinecone index and returns a PineconeVectorStore object.
        Initializes the index if it doesn't exist.
        """
        self._create_index_if_not_exists(dimension=dimension)
        index = self.pinecone.Index(self.index_name)
        vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
        print(f"Connected to Pinecone index: {self.index_name}")
        return vector_store

    def ingest_documents(self, documents: List[Document]):
        """
        Embeds and ingests documents into the Pinecone index.
        """
        try:
            vector_store = self.get_vector_store()
            vector_store.add_documents(documents)
            print(f"Successfully ingested {len(documents)} documents into Pinecone.")
        except Exception as e:
            print(f"Error ingesting documents to Pinecone: {e}")

    def delete_index(self):
        """
        Deletes the Pinecone index. Use with caution!
        """
        if self.index_name in self.pinecone.list_indexes().names():
            print(f"Deleting Pinecone index '{self.index_name}'...")
            self.pinecone.delete_index(self.index_name)
            print(f"Pinecone index '{self.index_name}' deleted.")
        else:
            print(f"Pinecone index '{self.index_name}' does not exist.")

# Example usage (for testing this module)
if __name__ == "__main__":

    config = Settings()
    processor = DocumentProcessor(file_path="app/data/HSC26-Bangla1st-Paper.pdf")
    processor.save_text_to_file()
    pinecone_manager = PineconeManager(config)

    # Ingest documents
    documents = processor.load_documents()
    chunks = processor.split_documents(documents)
    if chunks:
        pinecone_manager.ingest_documents(chunks)
    
    # Optionally, delete the index after testing
    # pinecone_manager.delete_index()