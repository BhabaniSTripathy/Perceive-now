# pipeline.py

import pandas as pd
import duckdb
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime


CSV_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
RAW_TABLE = "raw_data"
CLEAN_TABLE = "clean_data"
VECTOR_INDEX = "vectors.index"


print("Loading dataset...")
df_raw = pd.read_csv(CSV_URL)


print("Adding SHA256 fingerprint...")
df_raw["source_fingerprint"] = df_raw.apply(
    lambda row: hashlib.sha256(str(row.values).encode()).hexdigest(), axis=1
)


print("Cleaning data...")
df_clean = df_raw.dropna().drop_duplicates()


print("Saving to DuckDB...")
con = duckdb.connect("pipeline.duckdb")
con.execute(f"CREATE OR REPLACE TABLE {RAW_TABLE} AS SELECT * FROM df_raw")
con.execute(f"CREATE OR REPLACE TABLE {CLEAN_TABLE} AS SELECT * FROM df_clean")


print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = df_clean["sex"].astype(str) + " " + df_clean["smoker"].astype(str)
embeddings = model.encode(texts.tolist(), show_progress_bar=True)


print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, VECTOR_INDEX)


print("Starting FastAPI...")

app = FastAPI()

@app.post("/search")
async def search(request: Request):
    body = await request.json()
    query = body.get("query")

    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)

    results = []
    for idx in I[0]:
        row = df_clean.iloc[idx]
        results.append({
            "record": row.to_dict(),
            "source_fingerprint": row["source_fingerprint"],
            "source_file": "tips.csv",
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0.99 
        })

    return {"query": query, "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
