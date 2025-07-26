from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PINECONE_API_KEY: str
    GEMINI_API_KEY: str
    COHERE_API_KEY: str
    EMBEDDING_MODEL_NAME: str = "models/gemini-embedding-exp-03-07"
    PINECONE_INDEX_NAME: str = "my-rag-index-2"
    PINECONE_ENVIRONMENT: str = "aws"
    LLM_MODEL_NAME: str = "gemini-2.5-pro"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
