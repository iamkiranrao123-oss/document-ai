import re


def split_into_sentences(text):
    """
    Split text into sentences using common sentence-ending punctuation.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_long_sentence(sentence, chunk_size):
    """
    Split a sentence that is longer than chunk_size
    into smaller pieces without losing any text.
    """

    words = sentence.split()

    pieces = []
    current_piece = ""

    for word in words:

        candidate = (
            current_piece + " " + word
        ).strip()

        if len(candidate) <= chunk_size:
            current_piece = candidate

        else:
            if current_piece:
                pieces.append(current_piece)

            current_piece = word

    if current_piece:
        pieces.append(current_piece)

    return pieces


def chunk_pages(pages):
    """
    Split PDF pages into sentence-aware overlapping chunks.

    Each chunk:
    - stays within the configured chunk size whenever possible
    - keeps sentences intact
    - uses sentence-level overlap
    - preserves the original page number
    - receives a globally unique chunk ID
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

        # Break very long sentences into smaller pieces.
        processed_sentences = []

        for sentence in sentences:

            if len(sentence) <= chunk_size:
                processed_sentences.append(sentence)

            else:
                processed_sentences.extend(
                    split_long_sentence(
                        sentence,
                        chunk_size
                    )
                )

        current_sentences = []
        current_length = 0

        for sentence in processed_sentences:

            additional_length = len(sentence)

            if current_sentences:
                additional_length += 1

            candidate_length = (
                current_length + additional_length
            )

            if candidate_length <= chunk_size:

                current_sentences.append(sentence)
                current_length = candidate_length

            else:

                if current_sentences:

                    chunk_text = " ".join(
                        current_sentences
                    )

                    chunks.append({
                        "chunk_id": global_chunk_id,
                        "page_number": page_number,
                        "text": chunk_text
                    })

                    global_chunk_id += 1

                # Create sentence-level overlap.
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):

                    sentence_length = (
                        len(previous_sentence)
                    )

                    if overlap_sentences:
                        sentence_length += 1

                    if (
                        overlap_length
                        + sentence_length
                        <= overlap
                    ):
                        overlap_sentences.insert(
                            0,
                            previous_sentence
                        )

                        overlap_length += (
                            sentence_length
                        )

                    else:
                        break

                current_sentences = (
                    overlap_sentences + [sentence]
                )

                current_length = len(
                    " ".join(current_sentences)
                )

        # Store the final chunk from the page.
        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append({
                "chunk_id": global_chunk_id,
                "page_number": page_number,
                "text": chunk_text
            })

            global_chunk_id += 1

    return chunks