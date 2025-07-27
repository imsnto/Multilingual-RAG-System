from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langdetect import detect

from app.config import settings 
from app.utils.vector_store import PineconeManager
from app.utils.rag_chain import RAGChainBuilder
from app.services.english_query_service import process_prompt

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., example="কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", description="The question to be answered", title="User Query")

class QueryResponse(BaseModel):
    query: str
    answer: str

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    lang = detect(request.query)
    
    query = request.query
    if lang == 'en':
        response = process_prompt(request.query)
        query = response['response']

    pinecone_manager = PineconeManager(settings)
    try:
        vector_store_instance = pinecone_manager.get_vector_store()
        rag_builder = RAGChainBuilder(vector_store=vector_store_instance, config=settings)
        rag_chain = rag_builder.build_rag_chain()


        response = rag_chain.invoke(query)
        if lang == 'en':
            response = process_prompt(response)['response']
        return QueryResponse(
            query=request.query, 
            answer=response
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not run RAG chain test. Ensure Pinecone index is populated and API keys are correct.")
