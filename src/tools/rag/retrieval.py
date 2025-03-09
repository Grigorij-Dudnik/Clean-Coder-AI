import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from src.utilities.llms import init_llms_mini
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")
collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"


def get_collection():
    chroma_client = chromadb.PersistentClient(path=os.getenv('WORK_DIR') + '/.clean_coder/chroma_base')
    try:
        return chroma_client.get_collection(name=collection_name)
    except:
        # print("Vector database does not exist. (Optional) create it by running src/tools/rag/write_descriptions.py to improve file research capabilities")
        return False


def vdb_available():
    return True if get_collection() else False


def retrieve(question: str) -> str:
    """
    Retrieve files descriptions by semantic query.

    Parameters:
    question (str): The query to retrieve information for.

    Returns:
    str: A formatted response with file descriptions of found files.
    """
    collection = get_collection()
    retrieval = collection.query(query_texts=[question], n_results=8)
    
    # Use BinaryRanker to filter relevant documents
    binary_ranker = BinaryRanker()
    ranking_results = binary_ranker.rank(question, retrieval)

    # Filter documents that are marked as relevant (score = '1')
    response = ""
    for filename, score in ranking_results:
        if score == '1':
            # Find the corresponding document in the retrieval results
            idx = retrieval["ids"][0].index(filename)
            description = retrieval["documents"][0][idx]
            response += f"{filename}:\n\n{description}\n\n###\n\n"

    # If no relevant documents found, return a message
    if not response:
        return "No relevant documents found for your query."

    response += "\n\nRemember to see files before adding to final response!"
    return response


# New class added for binary ranking with lazy loading.
class BinaryRanker:
    """
    A binary document ranker that uses LLM to determine if a document is relevant.
    
    This class implements lazy loading of the LLM chain, meaning the chain
    is only initialized when the rank method is called. It evaluates whether
    each document is relevant to a given question, returning a binary score
    (0 or 1) for each document.
    """
    def __init__(self):
        """
        Initialize the BinaryRanker with lazy loading.
        
        The LLM chain is not created until the rank method is called.
        """
        # Lazy-loaded chain; not initialized until rank() is called.
        self.chain = None

    def initialize_chain(self):
        """
        Initialize the LLM chain if it hasn't been initialized yet.
        
        This method loads the prompt template from an external file, initializes the LLM,
        and builds the chain used for binary document ranking.
        """
        if self.chain is None:
            # Load the binary ranker prompt from an external file.
            grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            file_path = f"{grandparent_dir}/prompts/binary_ranker.prompt"
            with open(file_path, 'r') as file_handle:
                template_text = file_handle.read()
            prompt = ChatPromptTemplate.from_template(template_text)
            # Initialize LLMs with minimal intelligence and set run name to 'BinaryRanker'
            llms = init_llms_mini(tools=[], run_name='BinaryRanker')
            llm = llms[0].with_fallbacks(llms[1:])
            # Build the chain by combining the prompt template, the LLM instance, and StrOutputParser.
            self.chain = prompt | llm | StrOutputParser()

    def rank(self, question: str, retrieval: dict) -> list:
        """
        Rank documents based on their relevance to the question.

        Parameters:
        question (str): The query to evaluate document relevance against.
        retrieval (dict): The retrieval results from a vector database query.

        Returns:
        list: A list of tuples containing document IDs and their binary relevance scores ('0' or '1').
        """
        # Ensure the chain is initialized (lazy loading)
        self.initialize_chain()
        # Extract list of documents and their ids from the retrieval result.
        documents_list = retrieval["documents"][0]
        filenames_list = retrieval["ids"][0]
        # Build input for batch processing: list of dicts containing question, filename, and document.
        batch_inputs = []
        for idx, doc in enumerate(documents_list):
            batch_inputs.append({
                "question": question,
                "filename": filenames_list[idx],
                "document": doc
            })
        # Use the chain batch function to get binary outputs.
        results = self.chain.batch(batch_inputs)
        # Pair each document id with its binary ranking result.
        ranking = []
        for idx, result in enumerate(results):
            ranking.append((filenames_list[idx], result.strip()))
        return ranking


if __name__ == "__main__":
    # Example usage of BinaryRanker for testing.
    question = "Some tool that can change files"
    # Test the retrieve function
    results = retrieve(question)
    print("\n\n")
    print("results: ", results)
