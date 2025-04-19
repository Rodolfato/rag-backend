import argparse
import os
import time
from app.config import DOCUMENTS_PATH, vector_db_engine
from app.config import llm
from app.services.llm_services import query_llm
from app.utils.embedding_utils import (
    chunk_messages_with_context,
    extract_pdf_metadata,
    hash_documents,
    load_json,
    load_pdf_documents_subdirectories,
    make_chat_chunks_into_documents,
    split_documents_subdirectories,
    update_mongodb_with_links,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Carga la base de datos.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina los elementos dentro de la base de datos.",
    )
    parser.add_argument(
        "--load-msg",
        action="store_true",
        help="Carga la base de datos con mensajes de texto.",
    )
    parser.add_argument(
        "--query", type=str, help="Consulta a realizar sobre los datos."
    )
    parser.add_argument("--generate-pdf-json", type=str)
    parser.add_argument("--update-db-with-pdf-json", type=str)
    parser.add_argument(
        "--project-names",
        action="store_true",
        help="Obtiene los nombres de los proyectos.",
    )
    args = parser.parse_args()

    if args.load:
        overall_start_time = time.perf_counter()
        print("Cargando documentos almacenados en subdirectorios...")
        start_time = time.perf_counter()
        documents_directory = load_pdf_documents_subdirectories(DOCUMENTS_PATH)
        print(
            f"Documentos cargados, tiempo demorado: { time.perf_counter() - start_time} segundos."
        )
        print("Partiendo contenido en chunks...")
        start_time = time.perf_counter()
        chunks = split_documents_subdirectories(documents_directory, 512, 64)
        chunks_with_sha512 = hash_documents(chunks)
        print(
            f"Se crearon {len(chunks_with_sha512)} chunks. Tiempo demorado: { time.perf_counter() - start_time} segundos."
        )
        print("Cargando base de datos con chunks...")
        start_time = time.perf_counter()

        added_ids = vector_db_engine.load_db(chunks_with_sha512)

        print(
            f"Se agregaron {len(added_ids)} documentos. Tiempo demorado: { time.perf_counter() - start_time} segundos."
        )
        print(
            f"Tiempo total en el cargado de documentos: {time.perf_counter() - overall_start_time} segundos."
        )
    if args.load_msg:
        messages = load_json(file_path="app/data/messages/chat_history_each_msg.json")
        msg_chunks = chunk_messages_with_context(messages)
        msg_documents = make_chat_chunks_into_documents(msg_chunks)
        print(msg_documents)
        added_ids = vector_db_engine.load_db(msg_documents)
    if args.reset:
        print("Eliminando contenidos de base de datos")
        vector_db_engine.clear_db()
    if args.query:
        query_string = args.query
        print(f"La consulta es: {query_string}")
        """query_llm(
            vector_db_engine=vector_db_engine,
            query_text=query_string,
            model=llm,
            search_k=4,
        ) """
    if args.generate_pdf_json:
        directory = args.generate_pdf_json
        extract_pdf_metadata(directory)

    if args.update_db_with_pdf_json:
        directory = args.update_db_with_pdf_json
        update_mongodb_with_links(
            vector_db_engine.get_db_collection(),
            os.path.join(directory, "PDF File Names.json"),
        )
    if args.project_names:
        project_names = vector_db_engine.get_project_names()
        print(project_names)


if __name__ == "__main__":
    main()
