import argparse
from app.config import vector_db_engine
from app.config import llm
from app.services.llm_services import query_llm
from app.utils.embedding_utils import (
    chunk_messages_with_context,
    hash_documents,
    load_json,
    load_pdf_documents,
    make_chat_chunks_into_documents,
    split_documents,
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
    args = parser.parse_args()

    if args.load:
        print("Cargando base de datos")
        print("Cargando documentos...")
        documents = load_pdf_documents()
        print("Partiendo contenido en chunks...")
        chunks = split_documents(documents, 512, 64)
        chunks_with_sha512 = hash_documents(chunks)
        print("Cargando base de datos con chunks...")
        added_ids = vector_db_engine.load_db(chunks_with_sha512)
        print(f"Se agregaron {len(added_ids)} documentos")
    if args.load_msg:
        messages = load_json()
        msg_chunks = chunk_messages_with_context(messages)
        msg_documents = make_chat_chunks_into_documents(msg_chunks)
    if args.reset:
        print("Eliminando contenidos de base de datos")
        vector_db_engine.clear_db()
    if args.query:
        query_string = args.query
        print(f"La consulta es: {query_string}")
        query_llm(
            vector_db_engine=vector_db_engine,
            query_text=query_string,
            model=llm,
            search_k=4,
        )


if __name__ == "__main__":
    main()
