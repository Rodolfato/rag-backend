from pydantic import BaseModel


class Query(BaseModel):
    query_text: str
    search_k: int
