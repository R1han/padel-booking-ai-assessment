"""Chroma wrapper for semantic search over the prose.

Every failure path here returns None rather than raising, so retrieval can fall back to
SQLite FTS5 and the assistant keeps working when the vector store or the embedding
provider is unavailable.

Embeddings are supplied explicitly so Chroma never downloads its default ONNX model.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

log = logging.getLogger("padel.vectorstore")

COLLECTION = "padel"
_client = None
_collection = None


def _connect():
    global _client, _collection
    if _collection is not None:
        return _collection
    import chromadb

    cfg = settings()
    cfg.chroma_path.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(cfg.chroma_path))
    _collection = _client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    return _collection


def available() -> bool:
    try:
        return _connect().count() > 0
    except Exception as exc:  # noqa: BLE001 - degradation must never raise
        log.warning("chroma unavailable: %s", exc)
        return False


def rebuild(docs: list[dict[str, Any]], batch_size: int = 256) -> None:
    """Embed and index every prose document. Chunk ids stay internal; the metadata
    record_id is always the dataset's own ID so retrieval can report it verbatim."""
    from app import llm

    if not llm.has_credentials():
        log.warning("no API key configured; skipping embeddings (FTS5 still works)")
        return

    import chromadb

    cfg = settings()
    cfg.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.chroma_path))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - absent on a first run
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    embedder = llm.embeddings()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [f"{d['title']}\n{d['body']}" for d in batch]
        vectors = embedder.embed_documents(texts)
        collection.add(
            ids=[f"{d['record_id']}#{start + i}" for i, d in enumerate(batch)],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "record_id": d["record_id"],
                    "type": d["type"],
                    "branch_id": d["branch_id"] or "",
                    "title": d["title"],
                }
                for d in batch
            ],
        )
        log.info("embedded %d/%d", min(start + batch_size, len(docs)), len(docs))

    global _client, _collection
    _client, _collection = client, collection
    log.info("chroma: indexed %d documents", len(docs))


def search(
    query: str, k: int = 20, types: list[str] | None = None, branch_id: str | None = None
) -> list[dict] | None:
    """Ranked hits, or None if the vector store cannot answer -- the caller then degrades
    to lexical search rather than failing the request."""
    from app import llm

    if not llm.has_credentials():
        return None
    try:
        collection = _connect()
        if collection.count() == 0:
            return None
        vector = llm.embeddings().embed_query(query)

        clauses = []
        if types:
            clauses.append({"type": {"$in": types}})
        if branch_id:
            clauses.append({"branch_id": branch_id})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

        result = collection.query(query_embeddings=[vector], n_results=k, where=where)
    except Exception as exc:  # noqa: BLE001
        log.warning("vector search failed, falling back to lexical: %s", exc)
        return None

    hits = []
    documents = result.get("documents") or [[]]
    for i, (meta, distance) in enumerate(zip(result["metadatas"][0], result["distances"][0])):
        hits.append({
            "record_id": meta["record_id"],
            "type": meta["type"],
            "title": meta.get("title", ""),
            "score": 1.0 - distance,
            # The chunk that actually matched. A policy is one ~10KB record but many
            # chunks, so the parent record's opening is usually not the passage that
            # answered the query.
            "chunk": documents[0][i] if documents and documents[0] else "",
        })
    return hits
