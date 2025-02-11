"""Functions to create an index of files for RAG."""

import logging
import os
import sys
from pathlib import Path

import chromadb
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utilities.exceptions import MissingEnvironmentVariableError
from src.utilities.util_functions import join_paths
from src.utilities.llms import init_llms_mini


## Configure the logging level
logging.basicConfig(level=logging.INFO)


def relevant_extension(file_path: Path, file_extension_constraint: set[str]) -> bool:
    """Checker for whether file extension indicates a script."""
    # List of common code file extensions
    return file_path.suffix.lower() in file_extension_constraint


# read file content. place name of file in the top
def get_content(file_path: Path) -> str:
    """Collect file name and content to return them together as string."""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return file_path.name + "\n" + content


def evaluate_file(root: str, file: str, file_extension_constraint: set[str] | None, ignore: set[str]) -> Path | None:
    """Return file path if the file is to be considered."""
    file_path = Path(root).joinpath(file)
    if len(ignore.intersection(file_path.parts)) > 0:
        return None
    if file_extension_constraint and relevant_extension(
        file_path, file_extension_constraint=file_extension_constraint
    ):
        return file_path
    return None


def files_in_directory(
    directories_with_files_to_describe: list[str | Path],
    file_extension_constraint: set[str] | None,
    ignore: set[str],
) -> list[Path]:
    """Fetch paths of files in directory."""
    files_to_describe = []
    for directory in directories_with_files_to_describe:
        directory_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        tmp = [
            evaluate_file(
                root=str(directory),
                file=file,
                file_extension_constraint=file_extension_constraint,
                ignore=ignore,
            )
            for file in directory_files
        ]
        files_to_describe.extend(tmp)
        for root, _, files in os.walk(directory):
            tmp = [
                evaluate_file(
                    root=root,
                    file=file,
                    file_extension_constraint=file_extension_constraint,
                    ignore=ignore,
                )
                for file in files
            ]
            files_to_describe.extend(tmp)
    return files_to_describe


def save_file_description(file_path: Path, work_dir: str, description: str, file_description_dir: str) -> None:
    """Save file description."""
    file_name = file_path.relative_to(work_dir).as_posix().replace("/", "=")
    output_path = join_paths(file_description_dir, f"{file_name}.txt")
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(description)


def output_descriptions(
    files_to_describe: list[Path], chain: RunnableSequence, file_description_dir: str, work_dir: str
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
                work_dir=work_dir,
                description=description,
                file_description_dir=file_description_dir,
            )
            for file_path, description in zip(files_iteration, descriptions, strict=True)
        ]


def produce_descriptions(
    directories_with_files_to_describe: list[str | Path],
    file_description_dir: str,
    work_dir: str,
    ignore: set[str],
    file_extension_constraint: set[str] | None = None,
) -> None:
    """
    Produce short descriptions of files. Store the descriptions in .clean_coder folder in WORK_DIR.

    Inputs:
        directories_with_files_to_describe: directories from which files are to be described.
        file_description_dir: directory where generated file descriptions are to be saved to.
        work_dir: project directory worked on with Clean Coder.
        ignore: files and folders to ignore.
        file_extension_constraint: The list of file extension types accepted, if it's provided.

    Example:
        work_dir = os.getenv("WORK_DIR") # provide your own directory of choice if WORK_DIR is not set.
        if not work_dir:
            msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
            raise MissingEnvironmentVariableError(msg)
        file_description_dir = join_paths(work_dir, ".clean_coder/workdir_file_descriptions")
        file_extension_constraint = {
            ".js", ".jsx", ".ts", ".tsx", ".vue", ".py", ".rb", ".php", ".java", ".c", ".cpp", ".cs", ".go", ".swift",
            ".kt", ".rs", ".htm",".html", ".css", ".scss", ".sass", ".less", ".prompt",
        }
        ignore = {".clean_coder", ".coderrules"}
        produce_descriptions(directories_with_files_to_describe=[work_dir],
                        file_description_dir=file_description_dir,
                        work_dir=work_dir,
                        file_extension_constraint=file_extension_constraint,
                        ignore=ignore,
                        )
    """
    files_to_describe = files_in_directory(
        directories_with_files_to_describe=directories_with_files_to_describe,
        file_extension_constraint=file_extension_constraint,
        ignore=ignore,
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
    output_descriptions(
        files_to_describe=files_to_describe, work_dir=work_dir, chain=chain, file_description_dir=file_description_dir
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
    work_dir: str,
    file_description_dir: str,
    vdb_location: str = ".clean_coder/chroma_base",
) -> None:
    """
    Upload file descriptions to chroma database.

    Inputs:
        chroma_collection_name: name of the collection within Chroma vector database to save file descriptions in.
        file_description_dir: directory where generated file descriptions are available.
        work_dir: project directory worked on with Clean Coder.
        vdb_location: (optional) location for storing the vector database.

    Example:
        work_dir = os.getenv("WORK_DIR") # provide your own directory of choice if WORK_DIR is not set.
        if not work_dir:
            msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
            raise MissingEnvironmentVariableError(msg)
        file_description_dir = join_paths(work_dir, ".clean_coder/workdir_file_descriptions")
        file_extension_constraint = {
            ".js", ".jsx", ".ts", ".tsx", ".vue", ".py", ".rb", ".php", ".java", ".c", ".cpp", ".cs", ".go", ".swift",
            ".kt", ".rs", ".htm",".html", ".css", ".scss", ".sass", ".less", ".prompt",
        }
        ignore = {".clean_coder", ".coderrules"}
        produce_descriptions(directories_with_files_to_describe=[work_dir],
                        file_description_dir=file_description_dir,
                        work_dir=work_dir,
                        file_extension_constraint=file_extension_constraint,
                        ignore=ignore,
                        )
        chroma_collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"
        upload_descriptions_to_vdb(chroma_collection_name=chroma_collection_name, work_dir=work_dir, file_description_dir=file_description_dir)
    """
    chroma_client = chromadb.PersistentClient(path=join_paths(work_dir, vdb_location))
    collection = chroma_client.get_or_create_collection(
        name=chroma_collection_name,
    )

    # read files and upload to base
    upload_to_collection(collection=collection, file_description_dir=file_description_dir)


if __name__ == "__main__":
    # provide optionally which subfolders needs to be checked, if you don't want to describe all project folder
    # load environment
    load_dotenv(find_dotenv())
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
        raise MissingEnvironmentVariableError(msg)
    file_description_dir = join_paths(work_dir, ".clean_coder/workdir_file_descriptions")
    file_extension_constraint = {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".vue",
        ".py",
        ".rb",
        ".php",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".go",
        ".swift",
        ".kt",
        ".rs",
        ".htm",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".prompt",
    }
    ignore = {".clean_coder", ".coderrules"}
    produce_descriptions(
        directories_with_files_to_describe=[work_dir],
        file_description_dir=file_description_dir,
        work_dir=work_dir,
        file_extension_constraint=file_extension_constraint,
        ignore=ignore,
    )
    chroma_collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"
    upload_descriptions_to_vdb(
        chroma_collection_name=chroma_collection_name, work_dir=work_dir, file_description_dir=file_description_dir
    )
