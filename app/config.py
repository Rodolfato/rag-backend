import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from app.engines.chroma_engine import ChromaEngine
from app.engines.mongo_engine import MongoEngine
from app.utils.embedding_utils import get_jina_v2_embedding_function

load_dotenv(override=True)
# MONGODB_URI = os.getenv("MONGODB_URI")
# MONGODB_NAME = os.getenv("MONGODB_NAME")
# MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
CHROMA_PERSISTENT_DIRECTORY = os.getenv("CHROMA_PERSISTENT_DIRECTORY")

""" vector_db_engine = MongoEngine(
    conn_string=MONGODB_URI,
    db_name=MONGODB_NAME,
    collection=MONGODB_COLLECTION_NAME,
    search_index="default",
    search_index_function="cosine",
    embedding_model=get_jina_v2_embedding_function(),
) """

vector_db_engine = ChromaEngine(
    persist_directory=os.path.abspath(CHROMA_PERSISTENT_DIRECTORY),
    collection_name="documents",
)
print(os.path.abspath(CHROMA_PERSISTENT_DIRECTORY))

llm = OllamaLLM(model="llama3.2")
