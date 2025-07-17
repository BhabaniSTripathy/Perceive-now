# pipeline.py

import pandas as pd
import duckdb
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime

# === CONFIG ===
# ✅ A working, public CSV dataset (replace with your bigger one later)
CSV_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
RAW_TABLE = "raw_data"
CLEAN_TABLE = "clean_data"
VECTOR_INDEX = "vectors.index"

# === Load Data ===
print("Loading dataset...")
df_raw = pd.read_csv(CSV_URL)  # ✅ NO sep="\t"

# === Add SHA256 fingerprint ===
print("Adding SHA256 fingerprint...")
df_raw["source_fingerprint"] = df_raw.apply(
    lambda row: hashlib.sha256(str(row.values).encode()).hexdigest(), axis=1
)

# === Basic cleaning ===
print("Cleaning data...")
df_clean = df_raw.dropna().drop_duplicates()

# === Save to DuckDB ===
print("Saving to DuckDB...")
con = duckdb.connect("pipeline.duckdb")
con.execute(f"CREATE OR REPLACE TABLE {RAW_TABLE} AS SELECT * FROM df_raw")
con.execute(f"CREATE OR REPLACE TABLE {CLEAN_TABLE} AS SELECT * FROM df_clean")

# === Embedding ===
print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Pick a text column to embed — here using "sex" + "smoker" as an example
texts = df_clean["sex"].astype(str) + " " + df_clean["smoker"].astype(str)
embeddings = model.encode(texts.tolist(), show_progress_bar=True)

# === FAISS index ===
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, VECTOR_INDEX)

# === FastAPI Search ===
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
            "quality_score": 0.99  # Placeholder
        })

    return {"query": query, "results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)