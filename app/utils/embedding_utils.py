import os
import hashlib
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")


def load_pdf_documents(path: str = DOCUMENTS_PATH) -> List[Document]:
    """
    Carga documentos en formato PDF desde un directorio.

    Args:
        path (str): La ruta al directorio donde se encuentran los documentos. Por defecto, se utiliza la variable de entorno DOCUMENTS_PATH.

    Returns:
        List[Document]: Una lista de objetos Document que contienen la información de cada documento PDF.
    """
    document_loader = PyPDFDirectoryLoader(path)
    documents = document_loader.load()
    return documents


def split_documents(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Divide los documentos en fragmentos más pequeños (chunks) basados en un tamaño y overlap especificados.

    Args:
        documents (List[Document]): La lista de documentos a dividir.
        chunk_size (int): El tamaño máximo de cada fragmento.
        chunk_overlap (int): La cantidad de overlap entre los fragmentos.

    Returns:
        List[Document]: Una lista de documentos divididos en fragmentos (chunks).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    separated_documents = text_splitter.split_documents(documents)
    return separated_documents


def hash_content(content: str) -> str:
    """
    Genera un hash SHA-512 del contenido proporcionado.

    Args:
        content (str): El contenido en formato de texto que se va a hashear.

    Returns:
        str: El hash hexadecimal del contenido.
    """
    return hashlib.sha512(content.encode("utf-8")).hexdigest()


def hash_documents(documents: List[Document]) -> List[Document]:
    """
    Aplica una función de hash al contenido de cada documento y almacena el resultado en sus metadatos.

    Args:
        documents (List[Document]): La lista de documentos cuyos contenidos serán hasheados.

    Returns:
        List[Document]: La lista de documentos con los hashes agregados en la metadata bajo la clave 'page_content_sha512'.
    """
    hashed_documents = []
    for document in documents:
        document.metadata["page_content_sha512"] = hash_content(document.page_content)
        hashed_documents.append(document)
    return hashed_documents


def get_jina_v2_embedding_function():
    """
    Obtiene la función de embeddings utilizando el modelo 'jina/jina-embeddings-v2-base-es'.

    Returns:
        OllamaEmbeddings: El objeto de embeddings basado en el modelo especificado.
    """
    embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-es")
    return embeddings
