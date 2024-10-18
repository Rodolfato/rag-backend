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

    docs = vector_search(
        vector_store=vector_db_engine.init_vector_store(), query=query_text, k=search_k
    )

    page_contents = []
    for doc in docs:
        page_contents.append(doc.page_content)

    context_text = "\n\n".join(page_contents)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    model = model
    response_text = model.invoke(prompt)
    print(response_text)
    print(f"\n\n\nEl contexto para generar esta respuesta fue\n\n\n {context_text}")
    return response_text
