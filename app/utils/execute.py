import argparse
from app.engines.mongo_engine import MongoEngine
from app.services.llm_services import query_llm
from app.utils.embedding_utils import hash_documents, load_pdf_documents, split_documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Carga la base de datos.")
    parser.add_argument("--reset", action="store_true", help="Elimina los elementos dentro de la base de datos.")
    parser.add_argument("--generate", action="store_true", help="Genera y carga los vectores a la base de datos.")
    parser.add_argument("--query", type=str, help="Consulta a realizar sobre los datos.")
    args = parser.parse_args()
    vectorDBengine = MongoEngine

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
        vector_store = vectorDBengine.init_vector_store()
        added_ids = vectorDBengine.load_db(vector_store, chunks_with_sha512)
        print(f"Se agregaron {len(added_ids)} documentos")
    if args.reset:
        print("Eliminando contenidos de base de datos")
        vectorDBengine.clear_db()
    if args.query:
        query_string = args.query
        print(f"La consulta es: {query_string}")
        vector_store = vectorDBengine.init_vector_store()
        query_llm(vector_store=vector_store, query_text=query_string)
        

if __name__ == "__main__":
    main()