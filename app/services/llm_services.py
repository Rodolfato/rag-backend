from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from app.engines.engine_interface import Engine
from langchain_core.language_models import BaseLLM

PROMPT_TEMPLATE = ChatPromptTemplate(
    [
        (
            "system",
            "Eres un asistente cuya mision es responder preguntas en base a un contexto entregado. No inventes información.",
        ),
        (
            "system",
            "Si el contexto es un historial de mensajes, debes responder con la información entregada en dichos mensajes.",
        ),
        (
            "system",
            "La respuesta debe estar en texto plano. No utilices markdown ni ningun otro formato especial.",
        ),
        (
            "system",
            "El contexto para responder a la pregunta entregada por el usuario es el siguiente: {context}",
        ),
        (
            "human",
            "Utilizando unicamente la informacion entregada, responde a esta pregunta: {question}",
        ),
    ]
)


def query_llm(
    vector_db_engine: Engine,
    query_text: str,
    model: BaseLLM = OllamaLLM(model="llama3.2"),
    search_k: int = 4,
):

    query_text = query_text.lower()
    project_name = ""
    project_names = vector_db_engine.get_project_names()
    print(project_names)
    for project in project_names:
        if project in query_text:
            project_name = project
            print(f"Se encontro el nombre del proyecto en la query: {project_name}")
    if not project_name:
        full_response = {
            "model_response": "No se encontro el nombre del proyecto en la query",
            "sources": None,
        }
        return full_response

    vector_docs = vector_db_engine.vector_search(
        query=query_text,
        search_type="similarity",
        k=search_k,
        project_name=project_name,
    )
    keyword_docs = vector_db_engine.keyword_search(project_name, query_text, 5)
    docs = vector_docs + keyword_docs
    context_text = generate_context_text(docs)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    response_text = model.invoke(prompt)
    sources = generate_sources(docs)

    # sources_string = generate_sources_string(sources)  # Para debugging

    full_response = {
        "model_response": response_text,
        "sources": sources,
    }

    return full_response


def generate_sources(docs):
    sources = []
    for doc in docs:
        found_dict = next(
            (
                d
                for d in sources
                if d.get("title") == doc.metadata["title"].capitalize()
            ),
            None,
        )
        if found_dict:
            found_dict["pages"].append(doc.metadata["page"])
        else:
            doc_source = {
                "title": doc.metadata["title"].capitalize(),
                "pages": [doc.metadata["page"]],
                "author": doc.metadata["author"].title(),
                "link": doc.metadata["link"],
                "year": doc.metadata["year"],
            }
            sources.append(doc_source)
    for document in sources:
        document["pages"] = sorted(document["pages"])
    return sources


def generate_sources_string(sources):
    sources_string = ""
    for document in sources:
        sources_string += f"{document["title"]}, pag.{", ".join(map(str, document["pages"]))}. Autor: {document["author"]}\n"
    return sources_string


def generate_context_text(docs):
    page_contents = []
    for doc in docs:
        page_contents.append(doc.page_content)

    context_text = "\n\n".join(page_contents)
    return context_text
