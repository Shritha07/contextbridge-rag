
from typing import List

import cohere

from settings import Settings


def co_embed(
    texts: List[str],
    input_type: str = "search_document"
) -> List[List[float]]:
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list")

    if input_type not in {"search_document", "search_query"}:
        raise ValueError("Invalid embedding input_type")

    settings = Settings()

    client = cohere.Client(api_key=settings.cohere_key)

    response = client.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type=input_type
    )

    return response.embeddings
