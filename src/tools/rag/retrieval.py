import os
import chromadb
from chromadb.errors import NotFoundError
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from src.utilities.llms import init_llms_mini
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field



load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")
collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"

class BinaryRankingResult(BaseModel):
    """Structured output for binary document ranking. First analyze and provide reasoning, then make decision."""
    reasoning: str = Field(
        description="First, analyze the document and write 2-3 sentences explaining how well it matches the query and why"
    )
    is_relevant: bool = Field(
        description="Based on the reasoning above, make final decision if document is relevant (True) or not (False)"
    )


class BinaryRankingResult(BaseModel):
    """Structured output for binary document ranking. First analyze and provide reasoning, then make decision."""

    reasoning: str = Field(
        description="First, analyze the document and write 2-3 sentences explaining how well it matches the query and why"
    )
    is_relevant: bool = Field(
        description="Based on the reasoning above, make final decision if document is relevant (True) or not (False)"
    )


def get_collection():
    chroma_client = chromadb.PersistentClient(path=os.getenv("WORK_DIR") + "/.clean_coder/chroma_base")
    from chromadb.utils import embedding_functions

    # embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    # )
    try:
        return chroma_client.get_collection(name=collection_name)  # , embedding_function=embedding_function)
    except NotFoundError:
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

    # Filter documents that are marked as relevant (True)
    response = ""
    for filename, is_relevant in ranking_results:
        if is_relevant:
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
            with open(file_path, "r") as file_handle:
                template_text = file_handle.read()
            prompt = ChatPromptTemplate.from_template(template_text)

            # Initialize LLMs with minimal intelligence and set run name to 'BinaryRanker'
            llms = init_llms_mini(tools=[], run_name="BinaryRanker")
            llm = llms[0].with_fallbacks(llms[1:])

            # Configure LLM to use structured output
            llm_structured = llm.with_structured_output(BinaryRankingResult)

            # Build the chain with structured output
            self.chain = prompt | llm_structured


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
            batch_inputs.append({"question": question, "filename": filenames_list[idx], "document": doc})

        # Use the chain batch function to get structured outputs.
        results = self.chain.batch(batch_inputs)


        # Pair each document id with its binary ranking result.
        ranking = []
        for idx, result in enumerate(results):
            ranking.append((filenames_list[idx], result.is_relevant))

        return ranking


if __name__ == "__main__":
    # Example usage of BinaryRanker for testing.
    question = "Example of structured output of llm response."

    # Test the retrieve function
    results = retrieve(question)
    # print("\n\n")
    # print("results: ", results)

