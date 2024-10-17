from fastapi import APIRouter
from app.api.models import Query
from app.engines.mongo_engine import MongoEngine
from app.services.llm_services import query_llm

api_router = APIRouter()
vector_db_engine = MongoEngine

@api_router.get("/hello")
def say_hello():
    return {"message": "Hello from the RAG backend!"}

@api_router.post("/query/")
def ask_query(query: Query):
    model_response = query_llm(vector_store=vector_db_engine.init_vector_store, query_text=query.query_text, search_k=query.search_k)
    return {
        "message": "Query entregada con exito",
        "query": query,
        "model_response": model_response
    }
