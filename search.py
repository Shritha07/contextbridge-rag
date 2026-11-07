
import argparse
from typing import List, Dict, Any

import cohere
from sqlmodel import Session, select

from db_models import engine, Document
from embeddings import co_embed
from settings import Settings
from sqlalchemy import text


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    query_vector = co_embed(
        [query],
        input_type="search_query"
    )[0]

    distance = Document.embedding.cosine_distance(query_vector)

    candidates = {}

    with Session(engine) as session:

        # 1. Semantic vector search
        semantic_results = session.exec(
            select(Document, distance)
            .order_by(distance)
            .limit(limit)
        ).all()

        for rank, (document, score) in enumerate(
            semantic_results, start=1
        ):
            candidates[document.id] = {
                "document": document.document,
                "source": (document.meta or {}).get("source", "Unknown"),
                "distance": float(score),
                "hybrid_score": 0.0,
                "semantic_rank": rank,
                "keyword_rank": None
            }

        # 2. PostgreSQL full-text keyword search
        keyword_query = text("""
            SELECT
                id,
                document,
                meta,
                ts_rank_cd(
                    to_tsvector('english', coalesce(document, '')),
                    plainto_tsquery('english', :query)
                ) AS rank
            FROM document
            WHERE to_tsvector(
                'english',
                coalesce(document, '')
            ) @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        keyword_results = session.execute(
            keyword_query,
            {"query": query, "limit": limit}
        ).mappings().all()

        for rank, result in enumerate(keyword_results, start=1):
            document_id = result["id"]

            if document_id not in candidates:
                metadata = result["meta"] or {}

                candidates[document_id] = {
                    "document": result["document"],
                    "source": metadata.get("source", "Unknown"),
                    "distance": None,
                    "hybrid_score": 0.0,
                    "semantic_rank": None,
                    "keyword_rank": rank
                }
            else:
                candidates[document_id]["keyword_rank"] = rank

        # 3. Reciprocal Rank Fusion
        rrf_k = 60

        for item in candidates.values():
            if item["semantic_rank"] is not None:
                item["hybrid_score"] += (
                    1 / (rrf_k + item["semantic_rank"])
                )

            if item["keyword_rank"] is not None:
                item["hybrid_score"] += (
                    1 / (rrf_k + item["keyword_rank"])
                )

    # 4. Return combined ranking
    ranked_results = sorted(
        candidates.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return ranked_results[:limit]

def QA(query: str, limit: int = 5):
    """Generate an answer grounded in retrieved document context."""

    results = search(query, limit)

    if not results:
        print("No relevant documents found.")
        return

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {result['source']}\n"
            f"Content: {result['document']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are ContextBridge, a precise document question-answering assistant.

Answer the question using ONLY the supplied context.

STRICT RULES:
1. Every factual claim must be supported by the supplied context.
2. Cite the exact source number that directly supports each claim.
3. Never cite a source that does not explicitly support the claim.
4. Do not invent names, dates, relationships, or details.
5. If the context does not contain enough information, say so.
6. Keep the answer clear and concise.

CITATION FORMAT:
Use [Source 1], [Source 2], etc.
The source numbers must exactly match the numbered context sections.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    client = cohere.Client(api_key=Settings().cohere_key)

    response = client.chat(
        message=prompt,
        model="command-a-03-2025",
    )

    print("\n" + "=" * 60)
    print("CONTEXTBRIDGE ANSWER")
    print("=" * 60)
    print(response.text)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        distance = result["distance"]

        distance_label = (
            f"{distance:.4f}"
            if distance is not None
            else "N/A (keyword result)"
        )

        print(
            f"\n[Source {i}] {result['source']} "
            f"| cosine distance: {distance_label} "
            f"| hybrid score: {result['hybrid_score']:.5f}"
        )

        print(result["document"][:350])


def main():
    parser = argparse.ArgumentParser(
        description="ContextBridge semantic search and QA"
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question to ask your documents"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of document chunks to retrieve"
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    QA(" ".join(args.question), args.limit)


if __name__ == "__main__":
    main()
