from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings 
from app.utils.vector_store import PineconeManager
from app.utils.rag_chain import RAGChainBuilder


router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., example="কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", description="The question to be answered", title="User Query")

class QueryResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    print(request.query)
    pinecone_manager = PineconeManager(settings)
    try:
        vector_store_instance = pinecone_manager.get_vector_store()
        rag_builder = RAGChainBuilder(vector_store=vector_store_instance, config=settings)
        rag_chain = rag_builder.build_rag_chain()


        response = rag_chain.invoke(request.query)
        return QueryResponse(answer=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not run RAG chain test. Ensure Pinecone index is populated and API keys are correct.")
