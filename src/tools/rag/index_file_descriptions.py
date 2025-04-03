import os
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import chromadb
import sys
import questionary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utilities.util_functions import join_paths, read_coderrules
from src.utilities.start_work_functions import file_folder_ignored
from src.utilities.llms import init_llms_mini
from src.tools.rag.code_splitter import split_code
from src.utilities.print_formatters import print_formatted
from src.tools.rag.retrieval import vdb_available
from src.utilities.manager_utils import QUESTIONARY_STYLE
from tqdm import tqdm
import glob
from chromadb.utils import embedding_functions


load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")

GOLDEN = "\033[38;5;220m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Customize tqdm's bar format with golden and magenta colors
bar_format = (
    f"{GOLDEN}{{desc}}: {MAGENTA}{{percentage:3.0f}}%{GOLDEN}|"
    f"{{bar}}| {MAGENTA}{{n_fmt}}/{{total_fmt}} files "
    f"{GOLDEN}[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]{RESET}"
)


# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-mpnet-base-v2"
# )
# embedding_function = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
# )


def is_code_file(file_path):
    # List of common code file extensions
    code_extensions = {
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
    return file_path.suffix.lower() in code_extensions


# read file content. place name of file in the top
def get_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    content = file_path.name + "\n\n" + content
    return content


def collect_file_pathes(work_dir):
    """
    Collect and return a list of allowed code files from the given subfolders
    under the work_dir according to is_code_file criteria and .coderignore patterns.
    """
    allowed_files = []
    for root, _, files in os.walk(work_dir):
        for file in files:
            file_path = Path(root) / file
            if not is_code_file(file_path):
                continue
            relative_path_str = file_path.relative_to(work_dir).as_posix()
            if file_folder_ignored(relative_path_str):
                continue
            allowed_files.append(file_path)
    return allowed_files


def write_file_descriptions(file_list):
    """Writes descriptions of whole files in codebase. Gets list of files to describe and describes files in batches."""
    coderrules = read_coderrules()

    grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with open(f"{grandparent_dir}/prompts/describe_files.prompt", "r") as f:
        files_describe_template = f.read()
    prompt = ChatPromptTemplate.from_template(files_describe_template)

    llms = init_llms_mini(tools=[], run_name="File Describer")
    llm = llms[0].with_fallbacks(llms[1:])
    chain = prompt | llm | StrOutputParser()

    description_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")
    Path(description_folder).mkdir(parents=True, exist_ok=True)
    batch_size = 8
    pbar = tqdm(total=len(file_list), desc="[1/2]Describing files", bar_format=bar_format)

    for i in range(0, len(file_list), batch_size):
        files_iteration = file_list[i : i + batch_size]
        descriptions = chain.batch(
            [{"coderrules": coderrules, "code": get_content(file_path)} for file_path in files_iteration]
        )

        for file_path, description in zip(files_iteration, descriptions):
            file_name = file_path.relative_to(work_dir).as_posix().replace("/", "=")
            output_path = join_paths(description_folder, f"{file_name}.txt")

            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(description)

        # Update by actual number of files processed in this batch
        pbar.update(len(files_iteration))

    pbar.close()  # Don't forget to close the progress bar when done


def write_file_chunks_descriptions(file_list):
    """Writes descriptions of file chunks in codebase. Gets list of whole files to describe, divides files
    into chunks and describes each chunk separately."""

    coderrules = read_coderrules()

    grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with open(f"{grandparent_dir}/prompts/describe_file_chunks.prompt", "r") as f:
        chunks_describe_template = f.read()

    prompt = ChatPromptTemplate.from_template(chunks_describe_template)
    llms = init_llms_mini(tools=[], run_name="File Describer")
    llm = llms[0]
    chain = prompt | llm | StrOutputParser()

    description_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")
    Path(description_folder).mkdir(parents=True, exist_ok=True)

    # iterate chunks inside of the file
    for file_path in tqdm(file_list, desc="[2/2]Describing file chunks", bar_format=bar_format):
        file_content = get_content(file_path)
        # get file extenstion
        extension = file_path.suffix.lstrip(".")
        file_chunks = split_code(file_content, extension)
        # do not describe chunk of 1-chunk files
        if len(file_chunks) <= 1:
            continue
        descriptions = chain.batch(
            [{"coderrules": coderrules, "file_code": file_content, "chunk_code": chunk} for chunk in file_chunks]
        )

        for nr, description in enumerate(descriptions):
            file_name = f"{file_path.relative_to(work_dir).as_posix().replace('/', '=')}_chunk{nr}"
            output_path = join_paths(description_folder, f"{file_name}.txt")

            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(description)


def upload_descriptions_to_vdb():
    """Uploads descriptions, created by write_file_chunks_descriptions, into vector database."""
    print_formatted("Uploading file descriptions to vector storage...", color="magenta")
    chroma_client = chromadb.PersistentClient(path=join_paths(work_dir, ".clean_coder/chroma_base"))
    collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        # embedding_function=embedding_function
    )

    # read files and upload to base
    description_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")

    docs = []
    ids = []

    for root, _, files in os.walk(description_folder):
        for file in files:
            file_path = Path(root) / file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            docs.append(content)
            ids.append(file_path.name.replace("=", "/").removesuffix(".txt"))
            # upsert to vector storage by batches of 100
            if len(docs) >= 100:
                collection.upsert(documents=docs, ids=ids)
                # Clear the batch lists
                docs = []
                ids = []
    # upsert remaining docs
    collection.upsert(documents=docs, ids=ids)


def upsert_file_list(file_list):
    chroma_client = chromadb.PersistentClient(path=join_paths(work_dir, ".clean_coder/chroma_base"))
    collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        # embedding_function=embedding_function
    )

    descriptions_folder = join_paths(work_dir, ".clean_coder/files_and_folders_descriptions")

    docs = []
    ids = []
    print(f"file_list: {file_list}")
    # open every file starting with descriptions_folder/file and add content to list
    for file in file_list:
        pattern = os.path.join(descriptions_folder, f"{file.filename.replace('/', '=').removesuffix('.txt')}*")
        for file_path in glob.glob(pattern):
            with open(file_path, "r", encoding="utf-8") as file_content:
                content = file_content.read()
            file_path = Path(file_path)
            docs.append(content)
            ids.append(file_path.name.replace("=", "/").removesuffix(".txt"))
    print(f"ids: {ids}")
    collection.upsert(documents=docs, ids=ids)
    print_formatted("Re-indexing of modified files completed.", color="green")


def prompt_index_project_files():
    """
    Checks if the vector database (VDB) is available.
    If not, prompts the user via questionary to index project files for better search.
    Then asks if yous sure he want to do indexing. Then triggers write_and_index_descriptions().
    """
    if vdb_available():
        return
    answer = questionary.select(
        "Do you want to index your project files for improving file search?",
        choices=["Proceed", "Skip"],
        style=QUESTIONARY_STYLE,
        instruction="\nHint: Skip for testing Clean Coder; index for real projects.",
    ).ask()
    if answer == "Proceed":
        all_files = collect_file_pathes(work_dir)
        answer = questionary.select(
            f"Going to index {len(all_files)} files. Indexing could be time-consuming and costly. Are you ready to go?",
            choices=["Index", "Skip"],
            style=QUESTIONARY_STYLE,
            instruction="\nHint: Ensure you provided all files and directories you don't want to index in {WORK_DIR}/.clean_coder/.coderignore to avoid describing trashy files.",
        ).ask()
        if answer == "Index":
            write_and_index_descriptions(all_files)


def write_and_index_descriptions(file_list):
    # provide optionally which subfolders needs to be checked, if you don't want to describe all project folder
    write_file_descriptions(file_list)
    write_file_chunks_descriptions(file_list)
    upload_descriptions_to_vdb()


if __name__ == "__main__":
    # upload_descriptions_to_vdb()
    upsert_file_list(
        [
            "src/agents/debugger_agent.py",
        ]
    )
