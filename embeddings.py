import os
import argparse

import textwiz
from pypdf import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helpers import utils

DEFAULT_MODEL = 'SFR-Embedding-Mistral'

FAVRE = os.path.join(utils.BOOK_FOLDER, 'favre.pdf')


def load_and_clean_favre(path: str, write_truncated_pdf: bool = False) -> list[str]:
    """Load and clean the book "MATHÉMATIQUES & STATISTIQUES DE GESTION" by Jean-Pierre Favre.

    Parameters
    ----------
    path : str
        Path to the pdf file.
    write_truncated_pdf : bool, optional
        Whether to also write the truncated pdf file for easy later page mapping.

    Returns
    -------
    list[str]
        All pages we keep from the book.
    """

    # Load pdf and extract text
    reader = PdfReader(path)
    pages = [reader.pages[i].extract_text(orientations=0) for i in range(len(reader.pages))]
    # Remove first and last pages (index)
    pages = pages[12:802]

    if write_truncated_pdf:
        original_pages = reader.pages[12:802]
        original_pages = [original_page for original_page, text_page in zip(original_pages, pages) if text_page.strip() != '']

    # Remove empty pages
    pages = [page for page in pages if page.strip() != '']
    # Remove page headers
    pages = [utils.remove_header_favre(page) for page in pages]

    if write_truncated_pdf:
        writer = PdfWriter()
        for page in original_pages:
            writer.add_page(page)
        writer.write(os.path.join(utils.BOOK_FOLDER, 'favre_truncated.pdf'))

    return pages



def split_book(tokenizer, pages: list[str], page_separator: str = '\n', chunk_size: int = 1024,
               chunk_overlap: int = 256) -> list[str]:
    """Split the given `pages` from a book into chunks to embed.
    
    Parameters
    ----------
    tokenizer
        The tokenizer to use to compute the number of tokens of chunks.
    pages : list[str]
        The pages of the book (as extracted text).
    page_separator : str, optional
        How to separate the pages, by default '\n'
    chunk_size : int, optional
        The chunk size to use (number of tokens), by default 1024
    chunk_overlap : int, optional
        The chunk overlap to use (number of tokens), by default 256

    Returns
    -------
    list[str]
        The chunks.
    """

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size,
                                                                         chunk_overlap=chunk_overlap, keep_separator=True)
    full_book = page_separator.join(pages)
    chunks = splitter.split_text(full_book)

    return chunks



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



def embed_book(model: textwiz.HFEmbeddingModel, book_pages: list[str], book_name: str, page_separator: str = '\n',
               chunk_size: int = 1024, chunk_overlap: int = 256, max_batch_size: int = 20):
    """Embed the given book and save all results to disk.

    Parameters
    ----------
    model : textwiz.HFEmbeddingModel
        The model to use to create the embeddings.
    book_pages : list[str]
        The pdf pages (as text) of the book.
    book_name : str
        The name to give to the embedding folder.
    page_separator : str, optional
        Character used to separate pages, by default '\n'
    chunk_size : int, optional
        The chunk size to use (number of tokens), by default 1024
    chunk_overlap : int, optional
        The chunk overlap to use (number of tokens), by default 256
    max_batch_size : int, optional
        Maximum batch size to use to compute the embeddings, by default 20
    """
    
    # Compute the chunks
    chunks = split_book(model.tokenizer, book_pages, page_separator=page_separator, chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap)
    # Map chunks back to pdf pages
    chunk_pages = chunks_page_span(chunks, book_pages, page_separator=page_separator)

    # Compute the embedding of each chunk
    embeddings = model(chunks, max_batch_size=max_batch_size)

    # Create output folder if it does not exist already
    output_folder = os.path.join(utils.EMBEDDING_FOLDER, book_name)
    os.makedirs(output_folder, exist_ok=True)

    # Format the chunks and pages as dictionary to save as jsonl
    chunks_dic = [{'text': chunk} for chunk in chunks]
    page_mapping_dic = [{book_name: pages} for pages in chunk_pages]

    # Save embeddings, chunks, and chunks page mapping
    utils.save_npy(embeddings, os.path.join(output_folder, 'embeddings.npy'))
    utils.save_jsonl(chunks_dic, os.path.join(output_folder, 'chunks.jsonl'))
    utils.save_jsonl(page_mapping_dic, os.path.join(output_folder, 'chunks_page_mapping.jsonl'))



def main(chunk_size: int, chunk_overlap: int, page_separator: str = '\n', max_batch_size: int = 20):
    """Process all the books.

    Parameters
    ----------
    chunk_size : int
        _description_
    chunk_overlap : int
        _description_
    page_separator : str, optional
        _description_, by default '\n'
    max_batch_size : int, optional
        _description_, by default 20
    """

    book_names = ['favre']
    book_pages = [load_and_clean_favre(FAVRE, write_truncated_pdf=True)]

    # Load model
    model = textwiz.HFEmbeddingModel(DEFAULT_MODEL)

    for pages, name in zip(book_pages, book_names):
        embed_book(model, pages, name, page_separator=page_separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                   max_batch_size=max_batch_size)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create pdf embeddings')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size to embed (number of tokens).')
    parser.add_argument('--chunk_overlap', type=int, default=256, help='Overlap between chunks (number of tokens).')
    parser.add_argument('--batch_size', type=int, default=20, help='Max batch size to use during embeddings.')
    
    args = parser.parse_args()
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    max_batch_size = args.batch_size

    main(chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_batch_size=max_batch_size)