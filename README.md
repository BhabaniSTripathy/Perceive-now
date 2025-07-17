
##Overview

This prototype demonstrates an ingestion + traceability pipeline using a public **US Patents dataset (~30,000 rows)**.  
It shows how raw data is ingested, fingerprinted for traceability, embedded for semantic search, and served with an API.


**Pipeline Steps**

**Ingestion:**  
- Download public patents data (TSV).
- Store raw data and cleaned data in DuckDB.
- Add `source_fingerprint` column (SHA256 of each row).

**Embedding:**  
- Use HuggingFace **`all-MiniLM-L6-v2`** model to embed patent titles.
- Store vectors in a **FAISS** index for fast semantic search.

**Serving:**  
- Expose a **FastAPI** route `POST /search`:
  - Accepts a text query
  - Returns top 3 matching patents
  - Includes metadata: source file, timestamp, quality score.


##Schema Inference

- Uses **pandas** to infer schema from the TSV headers.
- Cleans rows with missing critical fields (`title`) and removes duplicates.


##Productionization

To productionize this:
- Orchestrate with **Airflow** or **Dagster** for scheduled ingestion & transforms.
- Use **Terraform** to manage cloud infra (S3 buckets, Snowflake, IAM roles).
- Store raw data in **S3**, use **Snowflake** or **DuckDB** for analytics.
- Host vector DB with **Weaviate** or **Pinecone** for scalable embeddings.
- Containerize with **Docker** and deploy FastAPI behind **API Gateway**.


##Run Locally

```bash
pip install -r requirements.txt
python pipeline.py
```

##Deliverables

- DuckDB file: `patents.duckdb`
- FAISS index: `patent.index`
- API server: FastAPI on `localhost:8000`

##Core Idea

This pipeline fingerprints, traces, and embeds data for zero-hallucination, auditable intelligence.

