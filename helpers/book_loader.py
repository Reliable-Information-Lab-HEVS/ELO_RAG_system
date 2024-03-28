import os
import re

from pypdf import PdfReader, PdfWriter

from helpers import utils

FAVRE = os.path.join(utils.BOOK_FOLDER, 'favre.pdf')
TILIERE = os.path.join(utils.BOOK_FOLDER, 'tiliere.pdf')


def load_process_and_resave(book: str, start_index: int, end_index: int, processor, save: bool = False) -> list[str]:
    """Handle the logic of loading a pdf book, extracting text, processing the text, and optionally saving truncated version
    to disk.

    Parameters
    ----------
    book : str
        Path to the pdf.
    start_index : int
        From which page number to start extracting text from.
    end_index : int
        From which page number to stop extracting text from.
    processor : _type_
        The function handling the processing of a page for the given `book`.
    save : bool, optional
        Whether to save a truncated version of the pdf to file or not, by default False

    Returns
    -------
    list[str]
        The text extracted from the book.
    """

    # Load pdf and extract text
    reader = PdfReader(book)

    # Keep only given pages
    original_pages = reader.pages[start_index:end_index]
    # Extract text
    text_pages = [page.extract_text(orientations=0) for page in original_pages]
    # Process pages
    text_pages = [processor(page) for page in text_pages]

    # Remove blank pages
    if save:
        original_pages = [original_page for original_page, text_page in zip(original_pages, text_pages) if text_page.strip() != '']
    text_pages = [page for page in text_pages if page.strip() != '']

    # Write truncated output
    if save:
        writer = PdfWriter()
        for page in original_pages:
            writer.add_page(page)
        writer.write(os.path.splitext(book)[0] + '_truncated.pdf')

    return text_pages


def process_page_favre(page: str) -> str:
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


def process_page_tiliere(page: str) -> str:
    """Remove the page header of the Tiliere book, as well as page numbers at the bottom.

    Parameters
    ----------
    page : str
        Page of the book.

    Returns
    -------
    str
        The processed page.
    """

    header_template = r'^Chapitre [0-9]\. .+'
    footer_template = r'\n[0-9]+$'

    page, removed_header = re.subn(header_template, '', page, count=1)
    page = re.sub(footer_template, '', page, count=1)

    # If we removed the header, check if we also need to strip the newline afterwards (not always the case)
    if removed_header == 1:
        page = re.sub(r'^\n', '', page, count=1)

    return page



def load_and_process_favre(save: bool = False) -> list[str]:
    return load_process_and_resave(FAVRE, 12, 802, process_page_favre, save=save)


def load_and_process_tiliere(save: bool = False) -> list[str]:
    return load_process_and_resave(TILIERE, 5, -1, process_page_tiliere, save=save)


LOADER = {
    'favre': load_and_process_favre,
    'tiliere': load_and_process_tiliere,
}