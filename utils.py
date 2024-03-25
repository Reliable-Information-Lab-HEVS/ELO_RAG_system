import re

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