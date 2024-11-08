import os
import re
import json
import uuid
import tempfile

import numpy as np
from pypdf import PdfReader, PdfWriter

# Path to the root
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))

DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

BOOK_FOLDER = os.path.join(DATA_FOLDER, 'books')

EMBEDDING_FOLDER = os.path.join(DATA_FOLDER, 'embeddings')


# Create temporary directory to store potential temporary pdf files that will be displayed
TEMPDIR = tempfile.TemporaryDirectory(dir=BOOK_FOLDER)


def validate_filename(filename: str, extension: str):
    """Check the validity of a filename and its extension. Create the path if needed.

    Parameters
    ----------
    filename : str
        The filename to check for.
    extension : str, optional
        The required extension for the filename, by default 'json'
    """

    # Extensions are always lowercase
    extension = extension.lower()

    dirname, basename = os.path.split(filename)

    # Check that the extension and basename are correct
    if basename == '':
        raise ValueError('The basename cannot be empty')
    
    if not basename.endswith(extension):
        raise ValueError(f'Filename must end with {extension}')

    # Make sure the path exists, and creates it if this is not the case
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_jsonl(dictionaries: list[dict], filename: str, append: bool = False):
    """Save a list of dictionaries to a jsonl file.

    Parameters
    ----------
    dictionaries : list[dict]
        The list of dictionaries to save.
    filename : str
        Filename to save the file.
    append : bool
        Whether to append at the end of the file or create a new one, default to False.
    """

    validate_filename(filename, extension='.jsonl')

    mode = 'a' if append else 'w'

    with open(filename, mode) as fp:
        for dic in dictionaries:
            fp.write(json.dumps(dic) + '\n')


def save_npy(data: np.ndarray, filename: str):
    """Save a numpy array to file.

    Parameters
    ----------
    data : np.ndarray
        The array to save.
    filename : str
        Filename to save the file.
    """

    validate_filename(filename, extension='.npy')
    np.save(filename, data, allow_pickle=False)


def load_jsonl(filename: str) -> list[dict]:
    """Load a jsonl file as a list of dictionaries.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    list[dict]
        The list of dictionaries.
    """

    dictionaries = []

    with open(filename, 'r') as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                dictionaries.append(json.loads(line))

    return dictionaries


def load_npy(filename: str) -> np.ndarray:
    """Load a npy file.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    np.ndarray
        The array.
    """

    return np.load(filename, allow_pickle=False)


def create_temporary_pdf(original_pdf_path: str, pages: list[int]) -> str:
    """Create a temporary pdf file corresponding to given `pages` of the original pdf.

    Parameters
    ----------
    original_pdf_path : str
        Path to the original pdf.
    pages : list[int]
        Indices of the pages to save.

    Returns
    -------
    str
        Filename corresponding to the truncated pdf.
    """

    # Load pdf and extract text
    reader = PdfReader(original_pdf_path)
    L = len(reader.pages)

    if not all([0 <= page < L for page in pages]):
        raise ValueError('Some pages are outside the range of original pdf')

    writer = PdfWriter()
    for page_index in pages:
        writer.add_page(reader.pages[page_index])

    unique_filename = os.path.join(TEMPDIR.name, str(uuid.uuid4()) + '.pdf')
    # Write pages
    writer.write(unique_filename)

    return unique_filename
