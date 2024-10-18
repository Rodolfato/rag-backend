from fastapi import APIRouter
from app.config import vector_db_engine
from app.config import llm
from app.api.models import Query
from app.services.llm_services import query_llm

api_router = APIRouter()


@api_router.get("/hello")
def say_hello():
    return {"message": "Hello from the RAG backend!"}


@api_router.post("/query/")
def ask_query(query: Query):
    model_response = query_llm(
        vector_db_engine=vector_db_engine,
        query_text=query.query_text,
        model=llm,
        search_k=query.search_k,
    )
    return {
        "message": "Query entregada con exito",
        "query": query,
        "model_response": model_response,
    }
