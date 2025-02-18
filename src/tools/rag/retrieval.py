"""Functions to retrieve the most relevant documents from an indexed RAG database."""
import os
from pathlib import Path

import chromadb
import cohere
from dotenv import find_dotenv, load_dotenv

from src.utilities.exceptions import MissingEnvironmentVariableError

load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")
if not work_dir:
    msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
    raise MissingEnvironmentVariableError(msg)
cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    cohere_client = cohere.Client(cohere_key)
collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"


def get_collection() -> bool | chromadb.PersistentClient:
    """Check if chroma database is available in WORK_DIR."""
    if cohere_key:
        chroma_client = chromadb.PersistentClient(path=os.getenv('WORK_DIR') + '/.clean_coder/chroma_base')
        try:
            return chroma_client.get_collection(name=collection_name)
        except:
            # print("Vector database does not exist. (Optional) create it by running src/tools/rag/write_descriptions.py to improve file research capabilities")
            return False
    return False


def vdb_available():
    return True if get_collection() else False


def retrieve(question: str) -> str:
    """Identifies the most relevant files that help answer a question."""
    # collection should be initialized once, in the class init
    collection = get_collection()
    retrieval = collection.query(query_texts=[question], n_results=8)
    reranked_docs = cohere_client.rerank(
        query=question,
        documents=retrieval["documents"][0],
        top_n=4,
        model="rerank-english-v3.0",
        #return_documents=True,
    )
    reranked_indexes = [result.index for result in reranked_docs.results]
    response = ""
    for index in reranked_indexes:
        filename = retrieval["ids"][0][index]
        description = retrieval["documents"][0][index]
        response += f"{filename}:\n\n{description}\n\n"
    response += "\n\nRemember to see files before adding to final response!"

    return response


if __name__ == "__main__":
    question = "Common styles, used in the main page"
    results = retrieve(question)
    print("\n\n")
    print("results: ", results)
