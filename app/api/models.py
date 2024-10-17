from pydantic import BaseModel

# Define a Pydantic model for the expected JSON body
class Query(BaseModel):
    query_text: str
    search_k: int