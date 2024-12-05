from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
from typing import List


class Engine(ABC):
    @abstractmethod
    def init_vector_store() -> VectorStore:
        """
        Inicializa la vector store utilizando los campos de clase indicados.

        Returns:
            VectorStore: La vector store inicializada.
        """
        pass

    @abstractmethod
    def load_db(documents: List[Document]) -> List[str]:
        """
        Carga documentos en la base de datos vectorial. Evita duplicados comparando los hashes de cada chunk.

        Args:
            vector_store (VectorStore): La vector store donde se cargarán los documentos.
            documents (List[Document]): La lista de documentos a agregar a la vector store.

        Returns:
            List[str]: La lista de IDs para los documentos agregados.
        """
        pass

    @abstractmethod
    def clear_db() -> None:
        """
        Elimina todos los documentos de una coleccion de la base de datos asociada con la vector store.
        """
        pass

    @abstractmethod
    def get_project_names() -> List[str]:
        """
        Obtiene una lista de los nombres de los proyectos desde la base de datos.

        Esta función ejecuta una operación de agregación en la colección de la base de datos
        para agrupar los documentos por el campo `project_name` y devolver una lista de valores
        únicos de dicho campo.

        El proceso se realiza mediante un pipeline de agregación de MongoDB que agrupa los documentos
        por `project_name` y luego proyecta solo los valores únicos.

        Args:
            self: Instancia de la clase que contiene la conexión y la colección de la base de datos.

        Returns:
            List[str]: Una lista de strings que contiene los nombres de los proyectos.

        Ejemplo:
            >>> project_names = instance.get_project_names()
            >>> print(project_names)
            ['proyecto1', 'proyecto2', 'proyecto3']
        """
        pass

    @abstractmethod
    def keyword_search(
        project_name: str, query: str, top_k: int = 10
    ) -> List[Document]:
        pass
