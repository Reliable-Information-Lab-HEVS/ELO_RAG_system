import os

import numpy as np

from helpers import utils

ALLOWED_EMBEDDINGS = tuple(dir for dir in os.listdir(utils.EMBEDDING_FOLDER) if not dir.startswith('.'))


def load_embedding(name: str) -> tuple[np.ndarray, list[str], list[list[int]]]:
    """Load all data corresponding to the embedding of a given document `name`.

    Parameters
    ----------
    name : str
        The name given to the document.

    Returns
    -------
    tuple[np.ndarray, list[str], list[list[int]]]
        The embeddings, chunks, and chunks page mapping.
    """

    if name not in ALLOWED_EMBEDDINGS:
        raise ValueError(f'The embedding name must be one of {*ALLOWED_EMBEDDINGS,}')
    
    path = os.path.join(utils.EMBEDDING_FOLDER, name)
    
    embeddings = utils.load_npy(os.path.join(path, 'embeddings.npy'))
    chunks = utils.load_jsonl(os.path.join(path, 'chunks.jsonl'))
    chunks_page_mapping = utils.load_jsonl(os.path.join(path, 'chunks_page_mapping.jsonl'))

    # Remove dictionaries used to save files
    chunks = [chunk['text'] for chunk in chunks]
    chunks_page_mapping = [mapping['pages'] for mapping in chunks_page_mapping]

    return embeddings, chunks, chunks_page_mapping