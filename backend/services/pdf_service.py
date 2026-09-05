from pypdf import PdfReader


def extract_pages(file_path):
    """
    Extract text from each PDF page separately.
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages