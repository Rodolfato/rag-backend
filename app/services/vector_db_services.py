from langchain.schema.document import Document
from langchain.vectorstores import VectorStore


def vector_search(
    vector_store: VectorStore,
    query: str,
    project_name: str = None,
    search_type: str = "similarity",
    k: int = 4,
) -> list[Document]:
    """
    Realiza una búsqueda en la vector store y filtra los resultados por el campo 'project_name' si se proporciona.

    Args:
        vector_store (VectorStore): Vector store a buscar.
        query (str): La consulta en lenguaje natural.
        project_name (str, opcional): El nombre del proyecto para filtrar los resultados. Si no se proporciona, no se filtra.
        search_type (str, opcional): El tipo de búsqueda. Por defecto, es "similarity".
        k (int, opcional): El número de documentos a devolver. Por defecto, es 4.

    Returns:
        List[Document]: Una lista de documentos que cumplen con la consulta de búsqueda y el filtro de 'project_name'.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(query)
    for doc in docs:
        if doc.metadata["project_name"] != project_name:
            docs.remove(doc)
    return docs
