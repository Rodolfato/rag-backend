### InTeractiOn Lab Knowledge Base

This knowledge base was created to serve as a centralized and searchable repository for the extensive academic documentation produced by the InTeractiOn Computer Science Lab at the University of Santiago of Chile. Much of this documentation was previously scattered across multiple sources without proper structure or indexing, making access and retrieval very difficult and inefficient.

The software is built using Retrieval-Augmented Generation (RAG) as its core technique and follows a modular design philosophy. This allows developers to integrate different embedding or generative models with minimal changes to the codebase by leveraging the LangChain library. Similarly, the system supports flexible integration of vector database engines. Currently, two engines are implemented as examples and use cases: MongoDB Atlas Vector and ChromaDB.

This project was undertaken as a final-year research and developing effort within an academic setting, supervised by a professor in an advisor capacity.

### Prerequisites
* Python 3.12
* [Spacy](https://spacy.io/usage)
* [Pip 24](https://pypi.org/project/pip/)

### Installation
#### On Linux
1. Clone the repo 
```sh
   git clone https://github.com/Rodolfato/rag-backend
```
2. Create a [python virtual environment](https://docs.python.org/3/library/venv.html)
```sh
   python  -m  venv  /path/to/new/virtual/environment
```
3. Install requirements
```sh
   pip install -r requirements.txt
```
4. Run on development mode
```sh
   uvicorn app.main:app --reload
```

### Using the built in CLI application to manage the database
Currently this softwre uses a Command Line Interface Application to manage the loading and vectorizing of PDF documents saved in the `/app/data/documents` directory. The available commands are as follows:

- **Load documents**: Loads all documents from the `/app/data/documents` directory, splits them into chunks, vectorizes each chunk, and adds the resulting embeddings to the vector database. Each subfolder name within `/app/data/documents` is used as the corresponding project name in the metadata of each chunk.
```sh
   python -m app.utils.execute --load
```
- **Reset database**: Deletes all documents inside the database.
```sh
   python -m app.utils.execute --reset
```
- **Load chat messages**: Loads chat history stored in the `app/data/messages/chat_history_each_msg.json` directory.
 ```sh
   python -m app.utils.execute --load-msg
```

### Future upgrades
- [ ] Automate the process of vectorizing and storing chat messages
- [ ] Replace the CLI app with a GUI for improved user experience
- [ ] Automate testing and deployment of the application
- [ ] Standardize a logging method for both console output and long-term text logs for easier debugging

