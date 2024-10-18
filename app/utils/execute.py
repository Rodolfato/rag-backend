import argparse
from app.config import vector_db_engine
from app.config import llm
from app.services.llm_services import query_llm
from app.utils.embedding_utils import (
    hash_documents,
    load_pdf_documents,
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
        "--generate",
        action="store_true",
        help="Genera y carga los vectores a la base de datos.",
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
        """ counter = 0
        for chunk in chunks_with_sha512:
            print(f"chunk number {counter}")
            print(chunk)
            print("\n\n\n")
            counter += 1 """
        print("Cargando base de datos con chunks...")
        added_ids = vector_db_engine.load_db(chunks_with_sha512)
        print(f"Se agregaron {len(added_ids)} documentos")
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
