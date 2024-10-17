import os
from typing import List
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from langchain.schema.document import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from app.engines.engine_interface import Engine
from app.utils.embedding_utils import get_jina_v2_embedding_function
from langchain_core.embeddings import Embeddings

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_NAME = os.getenv("MONGODB_NAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")

class MongoEngine(Engine):
    def init_vector_store(search_index: str="default", 
                        search_index_function: str="cosine", 
                        conn_string: str=MONGODB_URI,
                        db_name: str=MONGODB_NAME,
                        collection: str=MONGODB_COLLECTION_NAME,
                        embedding_model: Embeddings=get_jina_v2_embedding_function()
                        ) -> MongoDBAtlasVectorSearch:
        client = MongoClient(conn_string)
        mongodb_collection = client[db_name][collection]
        #Se crea la vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection=mongodb_collection,
            #Funcion que retorna el embedding a utilizar.
            #Aqui se puede cambiar el modelo a utilizar para realizar los embeddings a traves de LangChain.
            embedding=embedding_model,
            index_name=search_index,
            relevance_score_fn=search_index_function,
            text_key="page_content",
            embedding_key="page_content_embedding_jina_v2")
        return vector_store

    def load_db(vector_store: MongoDBAtlasVectorSearch, documents: List[Document]) -> List[str]:   
        added_ids = vector_store.add_documents(documents=documents)
        return added_ids

    def clear_db(self) -> None:
        collection = self.get_db_collection()
        deletions = collection.delete_many({}).deleted_count
        print(f"Se eliminaron {deletions} documentos.")

    def get_db(db_name: str, conn_str: str) -> Database:
        uri = conn_str
        client = MongoClient(uri)
        db = client[db_name]
        return db

    def get_db_collection(self, db_name: str = MONGODB_NAME, collection: str = MONGODB_COLLECTION_NAME, conn_str: str = MONGODB_URI) -> Collection:
        db = self.get_db(db_name, conn_str)
        collection = db[collection]
        return collection