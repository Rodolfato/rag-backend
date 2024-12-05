from typing import List
from chromadb import Client
from chromadb.config import Settings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from app.engines.engine_interface import Engine
from app.utils.embedding_utils import (
    check_all_documents_for_duplicate,
    get_jina_v2_embedding_function,
)
from langchain.embeddings.base import Embeddings
from app.utils.keyword_search_utils import preprocess_query_spacy, transform_to_document


class ChromaEngine(Engine):
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: Embeddings = get_jina_v2_embedding_function(),
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def init_vector_store(self) -> Chroma:
        chromadb_client = Client(Settings(persist_directory=self.persist_directory))
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            client=chromadb_client,
        )
        return vector_store

    def load_db(self, documents: List[Document]) -> List[str]:
        vector_store = self.init_vector_store()
        existing_documents = vector_store.similarity_search("", k=10000)
        final_documents = check_all_documents_for_duplicate(
            documents, existing_documents
        )

        print(
            f"Found {len(documents) - len(final_documents)} duplicates. Loading {len(final_documents)} documents."
        )

        added_ids = vector_store.add_documents(final_documents)
        return added_ids

    def clear_db(self) -> None:

        vector_store = self.init_vector_store()
        vector_store.delete_collection()
        print("Cleared all documents from the ChromaDB collection.")

    def get_project_names(self) -> List[str]:

        vector_store = self.init_vector_store()
        documents = vector_store.similarity_search("", k=10000)  # Fetch all documents

        project_names = list(
            {doc.metadata.get("project_name", "") for doc in documents}
        )
        return project_names

    def keyword_search(
        self, project_name: str, query: str, top_k: int = 10
    ) -> List[Document]:
        print(f"Query: {query}")
        keyword_query = preprocess_query_spacy(query)
        print(f"Processed query: {keyword_query}")
        if project_name in keyword_query:
            keyword_query = keyword_query.replace(project_name, "").strip()

        vector_store = self.init_vector_store()
        results = vector_store.similarity_search(
            keyword_query, k=top_k, filter={"project_name": project_name}
        )

        documents = [transform_to_document(result) for result in results]
        return documents
