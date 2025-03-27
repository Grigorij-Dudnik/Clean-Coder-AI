from src.tools.rag.index_file_descriptions import (
    write_file_descriptions,
    write_file_chunks_descriptions,
    upsert_file_list,
)


def update_descriptions(file_list):
    """
    Updates descriptions of provided files and rewrites them in vector storage.
    """
    write_file_descriptions(file_list)
    write_file_chunks_descriptions(file_list)

    # uploade file list descriptins to vdb
    upsert_file_list(file_list)
