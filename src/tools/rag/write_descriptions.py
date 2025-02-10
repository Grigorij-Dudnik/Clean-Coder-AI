"""Functions to create an index of files for RAG."""
import logging
import os
import sys
from pathlib import Path

import chromadb
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utilities.exceptions import MissingEnvironmentVariableError
from src.utilities.util_functions import join_paths
from src.utilities.llms import init_llms_mini


# load environment
load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")

## Configure the logging level
logging.basicConfig(level=logging.INFO)

def is_code_file(file_path: Path) -> bool:
    """Checker for whether file extension indicates a script."""
    # List of common code file extensions
    code_extensions = {
        ".js", ".jsx", ".ts", ".tsx", ".vue", ".py", ".rb", ".php", ".java", ".c", ".cpp", ".cs", ".go", ".swift",
        ".kt", ".rs", ".htm",".html", ".css", ".scss", ".sass", ".less", ".prompt",
    }
    return file_path.suffix.lower() in code_extensions


# read file content. place name of file in the top
def get_content(file_path) -> str:
    """Collect file name and content as string, together."""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return file_path.name + "\n" + content


def write_descriptions(subfolders_with_files: list[str | Path]) -> None:
    """Produce short descriptions of files. Store them in .clean_coder folder in WORK_DIR."""
    if not work_dir:
        msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
        raise MissingEnvironmentVariableError(msg)
    all_files = []
    for folder in subfolders_with_files:
        for root, _, files in os.walk(work_dir + str(folder)):
            for file in files:
                file_path = Path(root) / file
                if is_code_file(file_path):
                    all_files.append(file_path)

    prompt = ChatPromptTemplate.from_template(
    """Describe the following code in 4 sentences or less, focusing only on important information from integration point of view.
    Write what file is responsible for.\n\n'''\n{code}'''
    """,
    )
    llms = init_llms_mini(tools=[], run_name="File Describer")
    llm = llms[0]
    chain = prompt | llm | StrOutputParser()

    description_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")
    Path(description_folder).mkdir(parents=True, exist_ok=True)
    # iterate over all files, take 8 files at once
    batch_size = 8
    for i in range(0, len(all_files), batch_size):
        files_iteration = all_files[i:i + batch_size]
        descriptions = chain.batch([get_content(file_path) for file_path in files_iteration])
        logging.debug(descriptions)

        for file_path, description in zip(files_iteration, descriptions, strict=True):
            file_name = file_path.relative_to(work_dir).as_posix().replace("/", "=")
            output_path = join_paths(description_folder, f"{file_name}.txt")

            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(description)


def upload_descriptions_to_vdb() -> None:
    chroma_client = chromadb.PersistentClient(path=join_paths(work_dir, ".clean_coder/chroma_base"))
    collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
    )

    # read files and upload to base
    description_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")
    for root, _, files in os.walk(description_folder):
        for file in files:
            file_path = Path(root) / file
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            collection.upsert(
                documents=[
                    content,
                ],
                ids=[file_path.name.replace("=", "/").removesuffix(".txt")],
            )


if __name__ == "__main__":
    #provide optionally which subfolders needs to be checked, if you don't want to describe all project folder
    write_descriptions(subfolders_with_files=["/"])

    upload_descriptions_to_vdb()
