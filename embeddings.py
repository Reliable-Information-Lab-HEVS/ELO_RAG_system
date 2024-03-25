import os

import textwiz
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helpers import utils

DEFAULT_MODEL = 'SFR-Embedding-Mistral'

book = os.path.join(utils.BOOK_FOLDER, 'favre.pdf')


def load_and_clean_favre(path: str) -> list[str]:
    """Load and clean the book "MATHÉMATIQUES & STATISTIQUES DE GESTION" by Jean-Pierre Favre.

    Parameters
    ----------
    path : str
        Path to the pdf file.

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
    # Remove empty pages
    pages = [page for page in pages if page.strip() != '']
    # Remove page headers
    pages = [utils.remove_header_favre(page) for page in pages]

    return pages


def split_favre(pages: list[str], page_separator: str = '\n') -> list[str]:
    """Split the full book "MATHÉMATIQUES & STATISTIQUES DE GESTION" by Jean-Pierre Favre into chunks to 
    embed.
    
    Parameters
    ----------
    pages : list[str]
        The pages of the book we keep.
    page_separator : str, optional
        How to separate the pages, by default '\n'

    Returns
    -------
    list[str]
        The chunks.
    """

    tokenizer = textwiz.load_tokenizer(DEFAULT_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=1024,
                                                                         chunk_overlap=256, keep_separator=True)
    full_book = page_separator.join(pages)
    chunks = splitter.split_text(full_book)

    return chunks


def main():

    # Load and split the book
    pages = load_and_clean_favre(book)
    chunks = split_favre(pages, page_separator='\n')
    # TODO: find a way to map those processed pages to original pages, or truncate pdf to reflect them directly
    chunk_pages = utils.chunks_page_span(chunks, pages, page_separator='\n')

    # Load model
    model = textwiz.HFEmbeddingModel(DEFAULT_MODEL)

    # Compute the embedding of each chunk
    embeddings = model(chunks, max_batch_size=20)

    output_folder = os.path.join(utils.EMBEDDING_FOLDER, 'favre')
    os.makedirs(output_folder, exist_ok=True)

    chunks_dic = [{'text': chunk} for chunk in chunks]
    page_mapping_dic = [{'pages': pages} for pages in chunk_pages]

    # Save embeddings, chunks, and chunks page mapping
    utils.save_npy(embeddings, os.path.join(output_folder, 'embeddings.npy'))
    utils.save_jsonl(chunks_dic, os.path.join(output_folder, 'chunks.jsonl'))
    utils.save_jsonl(page_mapping_dic, os.path.join(output_folder, 'chunks_page_mapping.jsonl'))

    

if __name__ == '__name__':
    print('ojdcowa')

    main()