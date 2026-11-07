
import argparse
import logging
from hashlib import md5
from pathlib import Path

from sqlmodel import Session

from db_models import Document, engine
from embeddings import co_embed

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

BATCH_SIZE = 96


def chunk_text(text, size=1000, overlap=150):
    """Split text into overlapping character-based chunks."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")

    words = text.split()
    chunks = []
    current = []
    current_length = 0

    for word in words:
        added_length = len(word) + (1 if current else 0)

        if current and current_length + added_length > size:
            chunks.append(" ".join(current))

            # Retain trailing words for context overlap.
            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current):
                extra = len(previous_word) + (
                    1 if overlap_words else 0
                )
                if overlap_length + extra > overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += extra

            current = overlap_words
            current_length = len(" ".join(current))

        current.append(word)
        current_length += len(word) + (
            1 if len(current) > 1 else 0
        )

    # Do not lose the final chunk.
    if current:
        chunks.append(" ".join(current))

    return chunks


def index_document(file_path):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("Document is empty.")

    chunks = chunk_text(text)

    logging.info("Document: %s", path.name)
    logging.info("Generated %d chunks", len(chunks))

    with Session(engine) as session:
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            vectors = co_embed(batch)

            if len(vectors) != len(batch):
                raise RuntimeError("Embedding count mismatch.")

            for chunk, vector in zip(batch, vectors):
                document_id = md5(
                    f"{path.name}:{chunk}".encode("utf-8")
                ).hexdigest()

                document = Document(
                    id=document_id,
                    document=chunk,
                    embedding=vector,
                    meta={"source": path.name}
                )

                session.merge(document)

            session.commit()

            logging.info(
                "Indexed %d/%d chunks",
                min(start + len(batch), len(chunks)),
                len(chunks)
            )

    logging.info("Indexing completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="ContextBridge document indexing"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="rabin.txt",
        help="Path to the document to index"
    )

    args = parser.parse_args()
    index_document(args.file)


if __name__ == "__main__":
    main()
