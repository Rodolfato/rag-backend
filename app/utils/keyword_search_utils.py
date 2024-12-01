import re
import unicodedata
import spacy
from langchain.schema import Document
from langchain_core.language_models import BaseLLM


def preprocess_query(query: str) -> str:

    spanish_stopwords = {
        "de",
        "la",
        "que",
        "el",
        "en",
        "y",
        "a",
        "los",
        "se",
        "del",
        "las",
        "un",
        "por",
        "con",
        "no",
        "una",
        "su",
        "para",
        "es",
        "al",
        "como",
        "mas",
        "pero",
        "sus",
        "le",
        "ya",
        "o",
        "este",
        "si",
        "me",
        "sin",
        "sobre",
        "este",
        "este",
        "ser",
        "entre",
        "cuando",
        "todo",
        "tambien",
        "muy",
        "hasta",
        "aqui",
        "bien",
        "aquel",
        "cual",
        "ella",
        "esto",
        "ese",
        "solo",
        "algunos",
        "hacer",
        "o",
        "donde",
    }

    query = unicodedata.normalize("NFKD", query)
    query = "".join([c for c in query if not unicodedata.combining(c)])

    query = query.lower()
    words = re.findall(r"\b\w+\b", query)

    keywords = []
    for word in words:
        if word not in spanish_stopwords:
            keywords.append(word)

    return " ".join(keywords)


def preprocess_query_spacy(query: str) -> str:

    nlp = spacy.load("es_core_news_sm")

    doc = nlp(query)

    keywords = []

    for token in doc:
        if token.is_alpha and token.pos_ in {"NOUN", "PROPN"}:
            keywords.append(token.text.lower())

    return " ".join(keywords)


def transform_to_document(item_dict: dict) -> Document:

    page_content = item_dict.get("page_content", "")

    metadata = {key: value for key, value in item_dict.items() if key != "page_content"}

    return Document(page_content=page_content, metadata=metadata)
