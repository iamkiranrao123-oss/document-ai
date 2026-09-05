import re


def split_into_sentences(text):
    """
    Split text into sentences using common sentence-ending punctuation.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_pages(pages):
    """
    Split PDF pages into sentence-aware overlapping chunks.

    Each chunk keeps its original page number and
    receives a globally unique chunk ID.
    """

    chunk_size = 1000
    overlap = 200

    chunks = []
    global_chunk_id = 0

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        # Normalize whitespace.
        text = " ".join(text.split())

        if not text:
            continue

        sentences = split_into_sentences(text)

        current_chunk = ""

        for sentence in sentences:

            candidate = (
                current_chunk + " " + sentence
            ).strip()

            if len(candidate) <= chunk_size:

                current_chunk = candidate

            else:

                if current_chunk:

                    chunks.append({
                        "chunk_id": global_chunk_id,
                        "page_number": page_number,
                        "text": current_chunk
                    })

                    global_chunk_id += 1

                # Keep the last part of the previous chunk
                # as overlap for the next chunk.
                overlap_text = current_chunk[-overlap:]

                current_chunk = (
                    overlap_text + " " + sentence
                ).strip()

        # Store the final chunk from the page.
        if current_chunk:

            chunks.append({
                "chunk_id": global_chunk_id,
                "page_number": page_number,
                "text": current_chunk
            })

            global_chunk_id += 1

    return chunks