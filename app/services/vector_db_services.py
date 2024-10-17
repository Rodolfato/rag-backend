from langchain.schema.document import Document
from langchain.vectorstores import VectorStore

def vector_search(vector_store: VectorStore, query: str, search_type: str = "similarity", k: int = 4) -> list[Document]:
    retriever = vector_store.as_retriever(
        search_type = search_type,
        search_kwargs = { "k": k }
    )
    docs = retriever.invoke(query)
    return docs
