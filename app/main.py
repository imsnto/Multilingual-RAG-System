from fastapi import FastAPI

from app.routes.v1.endpoints import ask

app = FastAPI(
    title="Multilingual RAG System",
    version="1.0",
    description="A system that answers questions in multiple languages using a RAG model.",
)

app.include_router(ask.router, prefix="/api/v1")
