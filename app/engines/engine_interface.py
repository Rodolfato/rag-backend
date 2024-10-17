from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def init_vector_store():
        pass

    @abstractmethod
    def load_db():
        pass

    @abstractmethod
    def clear_db():
        pass