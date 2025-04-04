from src.tools.rag.index_file_descriptions import (
    write_file_descriptions,
    write_file_chunks_descriptions,
    upsert_file_list,
)
from src.utilities.objects import CodeFile
from src.utilities.print_formatters import print_formatted


def update_descriptions(file_list: [CodeFile]):
    """
    Updates descriptions of provided files and rewrites them in vector storage.
    """
    # TODO: remove old descriptions
    print_formatted("Updating descriptions...", color="magenta")
    write_file_descriptions(file_list)
    write_file_chunks_descriptions(file_list)

    # uploade file list descriptins to vdb
    upsert_file_list(file_list)
