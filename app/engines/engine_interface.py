from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
from typing import List


class Engine(ABC):
    @abstractmethod
    def init_vector_store() -> VectorStore:
        pass

    @abstractmethod
    def load_db():
        pass

    @abstractmethod
    def clear_db():
        pass

    @abstractmethod
    def get_project_names() -> List[str]:
        pass
