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


def remove_header_favre(page: str) -> str:
    """Remove the page header of the book "MATHÉMATIQUES & STATISTIQUES DE GESTION" by Jean-Pierre Favre.

    Parameters
    ----------
    page : str
        Page of the book.

    Returns
    -------
    str
        The page without the header.
    """

    try:
        first_line, remaining_page = page.split('\n', 1)
    # In case we do not find any new lign character -> not enough values to unpack error
    except ValueError:
        first_line = page
        remaining_page = ''

    odd_page_header_template = r'^[0-9]{0,3}– Mathématiques et statistiques de gestion'
    even_page_header_template = r'^Chapitre [0-9]{1,2} – .* – [0-9]{0,3}'

    if re.search(odd_page_header_template, first_line) or re.search(even_page_header_template, first_line):
        return remaining_page
    else:
        return first_line + '\n' + remaining_page


def chunks_page_span(chunks: list[str], pages: list[str], page_separator: str = '\n') -> list[list[int]]:
    """Find the document pages that each chunk in `chunks` spans. Indeed, usually a given chunk will span
    more than a given page. Obtaining this information is important to tell what pages are relevant to the
    user, and eventually display them.

    Parameters
    ----------
    chunks : list[str]
        The chunks spanning the entire document that we will embed.
    pages : list[str]
        The pages of the original document.
    page_separator : str, optional
        The separator used between each page to recreate the full book, by default '\n'

    Returns
    -------
    list[list[int]]
        For each chunk, this is a list of the page indices the chunk spans.
    """

    # Reconstruct the full book as a single string
    full_book = page_separator.join(pages)

    # Find index of the start of each page in the full book
    page_starts = []
    current_index = 0
    for page in pages:
        page_starts.append(current_index)
        current_index += len(page + page_separator)


    all_pages_span = []
    for chunk in chunks:

        # Find start and end of current chunk
        start = full_book.find(chunk)
        end = start + len(chunk)

        page_span = []
        for i in range(len(page_starts)-1):
            # This finds the index of the first page on which chunk is contained
            if page_starts[i] <= start and page_starts[i+1] > start:
                page_span.append(i)
                next = i + 1
                # Add all other pages that chunk may span
                while page_starts[next] < end:
                    page_span.append(next)
                    next += 1
                    # In case we are spanning last page, we need to stop
                    if next >= len(page_starts):
                        break

        # In this case the chunk is only contained on the last page
        if len(page_span) == 0 and page_starts[-1] <= start:
            page_span.append(len(page_starts)-1)

        all_pages_span.append(page_span)

    return all_pages_span


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
