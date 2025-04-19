import os
import shutil
from typing import List
from chromadb import PersistentClient
from langchain.schema import Document
from langchain_chroma import Chroma
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
        self.chromadb_client = PersistentClient(path=self.persist_directory)

    def init_vector_store(self) -> Chroma:

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            client=self.chromadb_client,
        )
        return vector_store

    def load_db(self, documents: List[Document]) -> List[str]:
        vector_store = self.init_vector_store()
        existing_documents = vector_store.get()
        sha512_dicts = []
        for metadata in existing_documents["metadatas"]:
            sha = metadata.get("page_content_sha512")
            if sha:
                sha512_dicts.append({"page_content_sha512": sha})

        final_documents = check_all_documents_for_duplicate(documents, sha512_dicts)

        print(
            f"Found {len(documents) - len(final_documents)} duplicates. Loading {len(final_documents)} documents."
        )
        if len(final_documents) > 0:
            added_ids = vector_store.add_documents(final_documents)
            return added_ids
        else:
            return []

    def clear_db(self) -> None:
        self.chromadb_client.delete_collection(name=self.collection_name)
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        print("Cleared all documents from the ChromaDB collection.")

    def get_project_names(self) -> List[str]:
        vector_store = self.init_vector_store()
        existing_data = vector_store.get()
        project_names = list(
            {
                metadata.get("project_name", "")
                for metadata in existing_data["metadatas"]
                if metadata.get("project_name", "")
            }
        )

        return project_names

    def keyword_search(
        self, project_name: str, query: str, top_k: int = 10
    ) -> List[Document]:
        print(f"Original query: {query}")

        keyword_query = preprocess_query_spacy(query)
        print(f"Processed query: {keyword_query}")

        vector_store = self.init_vector_store()

        try:
            results = vector_store.similarity_search(
                keyword_query, k=top_k, filter={"project_name": project_name}
            )
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

        if not results:
            print("No matching documents found.")

        return results

    def vector_search(
        self,
        query: str,
        project_name: str = None,
        search_type: str = "similarity",
        k: int = 4,
    ) -> list[Document]:
        print(f"Query: {query}")

        vector_store = self.init_vector_store()

        if search_type == "similarity":
            filter_criteria = {"project_name": project_name} if project_name else {}
            try:
                results = vector_store.similarity_search(
                    query, k=k, filter=filter_criteria
                )
            except Exception as e:
                print(f"Error during similarity search: {e}")
                return []

        return results
