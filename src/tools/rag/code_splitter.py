from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

code = """

"""
extension_to_language = {
    "cpp": "cpp",
    "go": "go",
    "java": "java",
    "kt": "kotlin",
    "js": "js",
    "jsx": "js",
    "vue": "js",
    "ts": "ts",
    "tsx": "ts",
    "mjs": "js",
    "cjs": "js",
    "php": "php",
    "proto": "proto",
    "py": "python",
    "rst": "rst",
    "rb": "ruby",
    "rs": "rust",
    "scala": "scala",
    "swift": "swift",
    "md": "markdown",
    "tex": "latex",
    "html": "html",
    "sol": "sol",
    "cs": "csharp",
    "cob": "cobol",
    "c": "c",
    "lua": "lua",
    "pl": "perl",
    "hs": "haskell",
    "ex": "elixir",
    "ps1": "powershell",
    "json": "json",
    "xml": "xml",
    "bash": "powershell",
    "zsh": "powershell",
    "sh": "powershell",
    "dockerfile": "proto",
}


def split_code(code: str, extension: str, chunk_size: int = 1000):
    """Splits code for smaller elements as functions. That allows to describe functions for semantic retrieval tool."""
    language = extension_to_language.get(extension)
    if not language:
        return []
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language(language), chunk_size=chunk_size, chunk_overlap=0
    )
    return splitter.split_text(code)


if __name__ == "__main__":
    splitted = split_code(code, "py")
    for doc in splitted:
        print(doc)
        print("###")
