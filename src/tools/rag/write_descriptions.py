"""Functions to create an index of files for RAG."""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utilities.exceptions import MissingEnvironmentVariableError
from src.utilities.llms import init_llms_mini
from src.utilities.start_work_functions import file_folder_ignored
from src.utilities.util_functions import join_paths

## Configure the logging level
logging.basicConfig(level=logging.INFO)

# load environment
load_dotenv(find_dotenv())  # load environment variables from .env file
work_dir = os.getenv("WORK_DIR")


def relevant_extension(file_path: Path, code_extensions: set[str]) -> bool:
    """Checker for whether file extension indicates a script."""
    # List of common code file extensions
    return file_path.suffix.lower() in code_extensions


# read file content. place name of file in the top
def get_content(file_path: Path) -> str:
    """Collect file name and content to return them together as string."""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return file_path.name + "\n" + content


def add_to_indexing_if_relevant(root: str, file: str, code_extensions: set[str] | None) -> Path | None:
    """Return file path if the file is to be considered."""
    file_path = Path(root).joinpath(file)
    if file_folder_ignored(str(file_path)):
        # ignore files and folders mentioned in .coderignore
        return None
    if not code_extensions:
        return file_path
    if relevant_extension(
        file_path, code_extensions=code_extensions,
    ):
        return file_path
    return None


def find_files_to_describe(
    directories_with_files_to_describe: list[str | Path],
    code_extensions: set[str] | None,
) -> list[Path]:
    """Fetch paths of files in directory."""
    files_to_describe = []
    for directory in directories_with_files_to_describe:
        directory_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        tmp = [
            add_to_indexing_if_relevant(
                root=str(directory),
                file=file,
                code_extensions=code_extensions,
            )
            for file in directory_files
        ]
        files_to_describe.extend(tmp)
        for root, _, files in os.walk(directory):
            tmp = [
                add_to_indexing_if_relevant(
                    root=root,
                    file=file,
                    code_extensions=code_extensions,
                )
                for file in files
            ]
            files_to_describe.extend(tmp)
    return files_to_describe


def save_file_description(file_path: Path, description: str, file_description_dir: str) -> None:
    """Save file description."""
    if not work_dir:
        msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
        raise MissingEnvironmentVariableError(msg)
    file_name = file_path.relative_to(work_dir).as_posix().replace("/", "=")
    output_path = join_paths(file_description_dir, f"{file_name}.txt")
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(description)


def describe_files(
    files_to_describe: list[Path], chain: RunnableSequence, file_description_dir: str,
) -> None:
    """Generate & output file descriptions to designated directory in WORK_DIR."""
    # iterate over all files, take 8 files at once
    batch_size = 8
    for i in range(0, len(files_to_describe), batch_size):
        files_iteration = [f for f in files_to_describe[i : i + batch_size] if f is not None]
        descriptions = chain.batch([get_content(file_path) for file_path in files_iteration])
        logging.debug(descriptions)
        [
            save_file_description(
                file_path=file_path,
                description=description,
                file_description_dir=file_description_dir,
            )
            for file_path, description in zip(files_iteration, descriptions, strict=True)
        ]


def produce_descriptions(
    directories_with_files_to_describe: list[str | Path],
    file_description_dir: str,
    code_extensions: set[str] | Literal["default"] | None = "default",
) -> None:
    """
    Produce short descriptions of files. Store the descriptions in .clean_coder folder in WORK_DIR.

    Inputs:
        directories_with_files_to_describe: directories from which files are to be described.
        file_description_dir: directory where generated file descriptions are to be saved to.
        ignore: files and folders to ignore.
        code_extensions: The list of file extension types accepted, if it's provided.
    """
    if code_extensions == "default":
        code_extensions = {".py", ".java", ".js", ".ts", ".html", ".css", ".scss", ".sql", ".json", ".xml"}
    files_to_describe = find_files_to_describe(
        directories_with_files_to_describe=directories_with_files_to_describe,
        code_extensions=code_extensions,
    )

    prompt = ChatPromptTemplate.from_template(
        """Describe the following code in 4 sentences or less, focusing only on important information from integration point of view.
    Write what file is responsible for.\n\n'''\n{code}'''
    """,
    )

    llms = init_llms_mini(tools=[], run_name="File Describer")
    llm = llms[0]
    chain = prompt | llm | StrOutputParser()
    Path(file_description_dir).mkdir(parents=True, exist_ok=True)
    describe_files(
        files_to_describe=files_to_describe, chain=chain, file_description_dir=file_description_dir
    )


def upload_to_collection(collection: chromadb.PersistentClient, file_description_dir: str) -> None:
    """Insert file information to chroma database."""
    for root, _, files in os.walk(file_description_dir):
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


def upload_descriptions_to_vdb(
    chroma_collection_name: str,
    file_description_dir: str,
    vdb_location: str = ".clean_coder/chroma_base",
) -> None:
    """
    Upload file descriptions to chroma database.

    Inputs:
        chroma_collection_name: name of the collection within Chroma vector database to save file descriptions in.
        file_description_dir: directory where generated file descriptions are available.
        vdb_location: (optional) location for storing the vector database.
    """
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
        raise MissingEnvironmentVariableError(msg)
    chroma_client = chromadb.PersistentClient(path=join_paths(work_dir, vdb_location))
    collection = chroma_client.get_or_create_collection(
        name=chroma_collection_name,
    )

    # read files and upload to base
    upload_to_collection(collection=collection, file_description_dir=file_description_dir)


if __name__ == "__main__":
    # provide optionally which subfolders needs to be checked, if you don't want to describe all project folder
    # load environment
    file_description_dir = join_paths(work_dir, ".clean_coder/workdir_file_descriptions")
    produce_descriptions(
        directories_with_files_to_describe=[work_dir],
        file_description_dir=file_description_dir,
    )
    chroma_collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"
    upload_descriptions_to_vdb(
        chroma_collection_name=chroma_collection_name, file_description_dir=file_description_dir,
    )
