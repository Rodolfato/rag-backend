from typing import List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langchain.schema.document import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from app.engines.engine_interface import Engine
from app.utils.embedding_utils import get_jina_v2_embedding_function
from langchain_core.embeddings import Embeddings


class MongoEngine(Engine):
    def __init__(
        self,
        conn_string: str,
        db_name: str,
        collection: str,
        search_index: str = "default",
        search_index_function: str = "cosine",
        embedding_model: Embeddings = get_jina_v2_embedding_function(),
    ):
        """
        Args:
            conn_string (str): String de conexion para la base de datos.
            db_name (str): El nombre de la base de datos.
            collection (str): El nombre de la coleccion dentro de la base de datos.
            search_index (str): Nombre del vector search index creado mediante la plataforma de Mongo Atlas Search. Default: "default".
            search_index_function (str): La función de búsqueda a utilizar (cosin, euclidean, dot product). Default: "cosine"
            embedding_model (Embeddings): El modelo de embeddings a utilizar.
        """
        self.conn_string = conn_string
        self.db_name = db_name
        self.collection = collection
        self.search_index = search_index
        self.search_index_function = search_index_function
        self.embedding_model = embedding_model

    def init_vector_store(self) -> MongoDBAtlasVectorSearch:
        """
        Inicializa la vector store utilizando los campos de clase indicados.

        Returns:
            MongoDBAtlasVectorSearch: La vector store inicializada.
        """
        client = MongoClient(self.conn_string)
        mongodb_collection = client[self.db_name][self.collection]

        vector_store = MongoDBAtlasVectorSearch(
            collection=mongodb_collection,
            embedding=self.embedding_model,
            index_name=self.search_index,
            relevance_score_fn=self.search_index_function,
            text_key="page_content",
            embedding_key="page_content_embedding_jina_v2",
        )

        return vector_store

    def load_db(self, documents: List[Document]) -> List[str]:
        """
        Carga documentos en el almacén vectorial.

        Args:
            vector_store (MongoDBAtlasVectorSearch): La vector store donde se cargarán los documentos.
            documents (List[Document]): La lista de documentos a agregar a la vector store.

        Returns:
            List[str]: La lista de IDs para los documentos agregados.
        """
        added_ids = self.init_vector_store().add_documents(documents=documents)
        return added_ids

    def clear_db(self) -> None:
        """
        Elimina todos los documentos de una coleccion MongoDB asociada con la vector store.
        """
        collection = self.get_db_collection()
        deletions = collection.delete_many({}).deleted_count
        print(f"Se eliminaron {deletions} documentos.")

    def get_db(self) -> Database:
        """
        Obtiene la instancia de la base de datos MongoDB.

        Returns:
            Database: La instancia de la base de datos MongoDB.
        """
        client = MongoClient(self.conn_string)
        db = client[self.db_name]
        return db

    def get_db_collection(self) -> Collection:
        """
        Obtiene la coleccion MongoDB de la base de datos.

        Returns:
            Collection: La instancia de la coleccion MongoDB.
        """
        db = self.get_db()
        collection = db[self.collection]
        return collection
