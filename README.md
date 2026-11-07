# ContextBridge

### Document Question Answering with Hybrid RAG

ContextBridge is a Python-based RAG application that transforms text documents into searchable knowledge and generates context-grounded answers using semantic retrieval, keyword search, and large language models.

**Python · Cohere · PostgreSQL · pgvector · RRF**

## Architecture

```text
User Query
    |
    v
Cohere Embeddings
    |
    +------------------+
    |                  |
    v                  v
Vector Search     Keyword Search
(pgvector)        (PostgreSQL)
    |                  |
    +--------+---------+
             |
             v
       RRF Ranking
             |
             v
      Retrieved Context
             |
             v
       Cohere LLM
             |
             v
       Final Answer
```

## Tech Stack

- **Language:** Python 3.11
- **Embeddings:** Cohere `embed-multilingual-v3.0`
- **LLM:** Cohere Command A
- **Database:** PostgreSQL 16 + pgvector
- **Keyword Search:** PostgreSQL Full-Text Search
- **Ranking:** Reciprocal Rank Fusion
- **Containerization:** Docker Compose

## Getting Started

```bash
git clone https://github.com/Shritha07/contextbridge-rag.git
cd contextbridge-rag

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Configure your Cohere API key and PostgreSQL credentials in `.env`.

Start the database:

```bash
docker compose up -d
```

Enable pgvector:

```bash
docker exec contextbridge-db psql -U contextbridge_user -d contextbridge -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Index a document and ask questions:

```bash
python index.py rabin.txt
python search.py "Who were Yitzhak Rabin's children?"
```

## Project Structure

```text
contextbridge-rag/
├── settings.py
├── db_models.py
├── embeddings.py
├── index.py
├── search.py
├── rabin.txt
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Validation

- PostgreSQL and pgvector connectivity verified.
- 38 document chunks indexed successfully.
- 1024-dimensional embeddings generated.
- Semantic and keyword retrieval tested.
- Hybrid ranking and LLM answer generation working.

*Development validation only; formal accuracy and performance benchmarks are future work.*

## Future Improvements

- PDF and multi-document support.
- Citation verification and retrieval evaluation.
- Web-based chat interface.
