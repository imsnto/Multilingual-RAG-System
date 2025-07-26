from multiprocessing import process
from pathlib import Path
import time
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DocumentProcessor:
    """
    Handles loading documents from a file and splitting them into chunks.
    """
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.file_path_txt = None 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    
    def save_text_to_file(self) -> None:
        self.file_path_txt = self.file_path.split(".")[0] + ".txt"
        text = self.extract_text_from_pdf()

        with open(self.file_path_txt, "w", encoding="utf-8") as f:
            f.write(text)

    def extract_text_from_pdf(self, lang='ben+eng') -> str:
        try:
            images = convert_from_path(self.file_path)
            extracted_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=lang)
                extracted_text += text + " "
            return extracted_text
        except Exception as e:
            return str(e)

    def load_documents(self) -> List[Document]:
        """
        Loads documents from the specified file path.
        """
        try:
            loader = TextLoader(self.file_path_txt, encoding="utf-8")
            documents = loader.load()
            print(f"Loaded {len(documents)} document(s) from {self.file_path}")
            return documents
        except Exception as e:
            print(f"Error loading document from {self.file_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks

# Example usage (for testing this module)
if __name__ == "__main__":
    processor = DocumentProcessor(file_path="app/data/HSC26-Bangla1st-Paper.pdf")
    processor.save_text_to_file()
    loaded_docs = processor.load_documents()

    
    if loaded_docs:
        chunks = processor.split_documents(loaded_docs)
        for i, chunk in enumerate(chunks[:3]): # Print first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            #print(chunk.page_content)
            #print(f"Metadata: {chunk.metadata}")