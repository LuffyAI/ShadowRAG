import hashlib
import numpy as np


async def normalize_query(q: str) -> str:
    return " ".join(q.lower().split())

async def query_hash(q: str, model: str) -> str:
    norm = await normalize_query(q)
    h = hashlib.sha256(f"{model}::{norm}".encode("utf-8")).hexdigest()
    return h

async def fetch_cached_query_embedding(conn, query, model="meta-llama/llama-3.1-8b-instruct", table="query_cache"):
    hash = await query_hash(query, model)
    sql = f"""
    SELECT embedding FROM {table}
    WHERE query_hash = ?;
    """
    cur = await conn.execute(sql, (hash,))
    row = await cur.fetchone()
    if row:
        emb_blob = row[0]
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        return embedding
    return None

async def store_cached_query_embedding(conn, query, embedding, model="meta-llama/llama-3.1-8b-instruct", table="query_cache"):
    
    if table == "query_cache":
        hash = await query_hash(query, model)
    else:
        hash = query

    emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()
    
    sql = f"""
    INSERT OR REPLACE INTO {table} (query_hash, embedding)
    VALUES (?, ?);
    """
    await conn.execute(sql, (hash, emb_blob))
    await conn.commit()

async def prepare_embedding_for_query(conn, embedding_func, query, model="meta-llama/llama-3.1-8b-instruct", table="query_cache"):
    cached_embedding = await fetch_cached_query_embedding(conn, query, model, table)

    if cached_embedding is not None:
        return cached_embedding.tobytes()
    else:
        embedding = await embedding_func(text=query)
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        await store_cached_query_embedding(conn, query, arr, model, table)
        return arr.tobytes()

