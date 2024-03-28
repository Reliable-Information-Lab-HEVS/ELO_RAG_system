import os

import numpy as np

from helpers import utils

ALLOWED_EMBEDDINGS = tuple(dir for dir in os.listdir(utils.EMBEDDING_FOLDER) if not dir.startswith('.'))


def load_embedding(name: str) -> tuple[np.ndarray, list[str], list[dict]]:
    """Load all data corresponding to the embedding of a given document `name`.

    Parameters
    ----------
    name : str
        The name given to the document.

    Returns
    -------
    tuple[np.ndarray, list[str], list[dict]]
        The embeddings, chunks, and chunks page mapping.
    """

    if name not in ALLOWED_EMBEDDINGS:
        raise ValueError(f'The embedding name must be one of {*ALLOWED_EMBEDDINGS,}')
    
    path = os.path.join(utils.EMBEDDING_FOLDER, name)
    
    embeddings = utils.load_npy(os.path.join(path, 'embeddings.npy'))
    chunks = utils.load_jsonl(os.path.join(path, 'chunks.jsonl'))
    chunks_page_mapping = utils.load_jsonl(os.path.join(path, 'chunks_page_mapping.jsonl'))

    # Remove dictionary used to save text chunks
    chunks = [chunk['text'] for chunk in chunks]

    return embeddings, chunks, chunks_page_mapping


def combine_embeddings(embeddings: list[np.ndarray], chunks: list[list[str]],
                       page_mapping: list[list[dict]]) -> tuple[np.ndarray, list[str], list[dict]]:
    """Combine all embeddings (concatenate the lists and vectors).

    Parameters
    ----------
    embeddings : list[np.ndarray]
        The different embeddings.
    chunks : list[list[str]]
        The different chunks.
    page_mapping : list[list[dict]]
        The different page mappings.

    Returns
    -------
    tuple[np.ndarray, list[str], list[dict]]
        Concatenation of all embeddings.
    """

    final_embeddings = np.concatenate(embeddings, axis=0)
    final_chunks, final_page_mapping = [], []
    for a, b in zip(chunks, page_mapping):
        final_chunks.extend(a)
        final_page_mapping.extend(b)

    return final_embeddings, final_chunks, final_page_mapping


def load_all_embeddings() -> tuple[np.ndarray, list[str], list[list[int]]]:
    """Load all data corresponding to ALL embeddings available on disk.

    Returns
    -------
    tuple[np.ndarray, list[str], list[list[int]]]
        The embeddings, chunks, and chunks page mapping.
    """

    embeddings, chunks, page_mappings = [], [], []

    for name in ALLOWED_EMBEDDINGS:
        embed, texts, mapping = load_embedding(name)
        embeddings.append(embed)
        chunks.append(texts)
        page_mappings.append(mapping)

    return combine_embeddings(embeddings, chunks, page_mappings)
