import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
import logging
from logging.handlers import RotatingFileHandler


load_dotenv(override=True)
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
MESSAGES_PATH = os.getenv("MESSAGES_PATH")

# TODO MOVE THIS LOGGING INTO A NEW FILE CALLED LOGGER_CONFIG.PY
# TODO Also make a new file name each time it runs
# TODO utilizar el logger para el funcionamiento de la aplicacion
# TODO agregar logger a la consola
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler and set the formatter
file_handler = RotatingFileHandler("logfile.log", backupCount=1)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


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


def load_pdf_documents_subdirectories(path: str) -> List[List[Document]]:
    subdir_documents = []

    for subdir, _, _ in os.walk(path):
        if subdir == path:
            continue

        json_path = os.path.join(subdir, "PDF File Names.json")
        links_data = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as json_file:
                links_data = json.load(json_file)

        documents = load_pdf_documents(subdir)
        for docu in documents:

            filename_parts = docu.metadata["source"].split("/")[-1].split("_")
            docu.metadata["project_name"] = subdir.split("/")[-1]
            docu.metadata["author"] = filename_parts[1].replace("-", " ").lower()
            docu.metadata["year"] = filename_parts[0]

            # Se busca el link y titulo del documento en el JSON utilizando la fecha y el autor
            matching_entry = next(
                (
                    entry
                    for entry in links_data
                    if entry["year"] == docu.metadata["year"]
                    and entry["author"].lower() == docu.metadata["author"].lower()
                ),
                None,
            )
            if matching_entry:
                docu.metadata["title"] = matching_entry["title"]
                docu.metadata["link"] = matching_entry["link"]
                # El PDF loader empieza a contar las paginas desde 0. Se le suma 1.
                docu.metadata["page"] = docu.metadata["page"] + 1
            else:
                print(
                    f"Documento NO encontrado: {docu.metadata}. Revisar que el autor, fecha y titulo sea igual entre el JSON y el nombre del archivo del documento."
                )

        subdir_documents.append(documents)

    return subdir_documents


def split_documents_subdirectories(
    documents: List[List[Document]], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    chunks = []
    for directory in documents:
        dir_chunks = split_documents(directory, chunk_size, chunk_overlap)
        chunks.extend(dir_chunks)
    return chunks


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


def load_json(file_path: str = MESSAGES_PATH) -> List[Dict]:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def chunk_messages_with_context(
    messages: List[Dict], window_size: int = 3
) -> List[Dict]:
    messages_sorted = sorted(messages, key=lambda x: (x["chat_id"], x["timestamp"]))

    chunks = []

    for i in range(len(messages_sorted)):
        msg = messages_sorted[i]
        print(f"\t\tCreando chunk con el mensaje {msg}\n")
        same_chat_messages = []
        print(f"Los mensajes pertenecientes a este mismo chat son:\n")
        for m in messages_sorted:
            if m["chat_id"] == msg["chat_id"]:
                print(f"\t{m}")
                same_chat_messages.append(m)

        k = same_chat_messages.index(msg)
        start_index = max(0, k - window_size)
        end_index = min(len(same_chat_messages), k + window_size + 1)
        print(
            f"\nEl mensaje esta en la posicion {k}. La ventana sera {start_index}, {end_index}\n"
        )

        context_messages = []
        for j in range(start_index, end_index):
            print(f"Contexto encontrado en el mensaje {j}: {same_chat_messages[j]}\n")
            context_messages.append(same_chat_messages[j])

        chunk = {
            "chat_id": msg["chat_id"],
            "subject": msg["subject"],
            "central_message_id": msg["message_id"],
            "central_text": msg["text"],
            "centra_message_sender_id": msg["sender_id"],
            "timestamp": msg["timestamp"],
            "context": context_messages,
        }

        print("Finalmente el chunk queda de la siguiente forma:")
        for key, value in chunk.items():
            if key != "context":
                print(f"\t{key}: {value}")
        for msg in chunk["context"]:
            print(f"\t\t{msg}")

        chunks.append(chunk)
        print("\n\n")

    return chunks


def make_chat_chunks_into_documents(messages: List[Dict]) -> List[Document]:
    documents = []
    for message in messages:

        chunk_text = ""
        for msg in message["context"]:
            chunk_text += msg["sender_id"] + ": " + msg["text"] + "\n"

        document = Document(
            page_content=chunk_text.strip(),
            metadata={
                "chat_id": message["chat_id"],
                "subject": message["subject"],
                "timestamp": message["timestamp"],
                "central_text": message["central_text"],
            },
        )
        documents.append(document)
    return documents


def check_all_documents_for_duplicate(lg_documents, db_documents):
    """
    Verifica si los documentos en `lg_documents` ya existen en `db_documents` basándose en el hash del contenido (`page_content_sha512`).
    Registra los documentos nuevos y los duplicados, y devuelve solo los documentos nuevos.

    Esta función compara los hashes de los documentos en `lg_documents` con los documentos existentes en `db_documents`.
    Si se encuentra un documento duplicado, se agrega a la lista `duplicates`. Si es un documento nuevo, se agrega a la lista `new_documents` y se registra una advertencia.

    Args:
        lg_documents (list): Lista de documentos nuevos a verificar.
        db_documents (list): Lista de documentos ya existentes en la base de datos.

    Returns:
        list: Lista de documentos nuevos que no están duplicados.

    Registra:
        - Información sobre documentos duplicados y nuevos.
        - Si falta el hash en un documento, se registra un error.
        - Registra el total de documentos duplicados y nuevos encontrados.
    """
    db_hashes = {doc.get("page_content_sha512") for doc in db_documents}
    duplicates = []
    new_documents = []

    for lg_document in lg_documents:
        lg_hash = lg_document.metadata.get("page_content_sha512")
        if not lg_hash:
            logger.error("No se encontro el hash en el documento.")
            continue

        if lg_hash in db_hashes:
            duplicates.append(lg_document)
        else:
            new_documents.append(lg_document)
            logger.info(f"Nuevo documento encontrado:")
            logger.info(f"\n\tProject: {lg_document.metadata.get('project_name')}")
            logger.info(f"\n\tTitle: {lg_document.metadata.get('title')}")

    logger.info(f"Total documentos duplicados: {len(duplicates)}.")
    logger.info(f"Total documentos nuevos: {len(new_documents)}.")
    return new_documents


def extract_pdf_metadata(pdf_directory):
    pdf_metadata = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            parts = filename[:-4].split("_")
            if len(parts) >= 3:
                year = parts[0]
                author = parts[1].replace("_", " ")
                title = "_".join(parts[2:]).replace("_", " ")

                pdf_metadata.append(
                    {
                        "year": year,
                        "author": author,
                        "title": title,
                        "link": "",
                    }
                )

    output_path = os.path.join(pdf_directory, "PDF File Names.json")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(pdf_metadata, json_file, ensure_ascii=False, indent=4)

    print(f"Metadata extraida y guardada en {output_path}")


def update_mongodb_with_links(collection, json_file):
    # TODO TRANSFORMAR ESTO A UN UPDATE EN BASE AL JSON Y MOVERLO AL ENGINE
    with open(json_file, "r", encoding="utf-8") as file:
        links_data = json.load(file)

    for document in collection.find():
        matching_entry = next(
            (
                entry
                for entry in links_data
                if entry["year"] == document.get("year")
                and entry["author"].lower() == document.get("author", "").lower()
                and entry["title"].lower() == document.get("title", "").lower()
            ),
            None,
        )

        if matching_entry:
            document["link"] = matching_entry["link"]

            collection.update_one(
                {"_id": document["_id"]}, {"$set": {"link": matching_entry["link"]}}
            )
            print(
                f"Se actualizo el documento ({document['author'], document['year']}) con link: {matching_entry['link']}"
            )

    print("Todos lo documentos procesados.")
