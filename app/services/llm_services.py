from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from app.engines.engine_interface import Engine
from app.services.vector_db_services import vector_search
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
) -> str:

    query_text = query_text.lower()
    project_name = ""
    project_names = vector_db_engine.get_project_names()
    print(project_names)
    for project in project_names:
        if project in query_text:
            project_name = project
            print(f"Se encontro el nombre del proyecto en la query: {project_name}")
    if not project_name:
        response_text = f"No se proporcionó el nombre del proyecto en la consulta.\nLos proyectos disponible para consultar son:\n{", ".join(project_names)}."
        return response_text

    docs = vector_search(
        vector_store=vector_db_engine.init_vector_store(),
        query=query_text,
        search_type="similarity",
        k=search_k,
        project_name=project_name,
    )

    page_contents = []
    for doc in docs:
        page_contents.append(doc.page_content)

    context_text = "\n\n".join(page_contents)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    model = model
    response_text = model.invoke(prompt)
    sources = "\n"
    for doc in docs:
        sources += f"{doc.metadata["title"].capitalize()}, pag.{doc.metadata["page"]}. {doc.metadata["author"]}\n\n"
    print(response_text)
    full_response = f"{response_text}\n\nFuentes: {sources}"
    print(
        f"\n\n\nEl contexto para generar la respuesta a esta pregunta: {query_text} fue:\n\n\n {context_text}"
    )
    for doc in docs:
        print(doc.metadata)
        print()
        print()
    return full_response
