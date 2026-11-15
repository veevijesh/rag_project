"""Embeds the policy file, stores it in Qdrant, and runs role-filtered searches."""
import hashlib
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from chunker import load_and_chunk

POLICY_FILE = Path(__file__).resolve().parent / "company_policy.txt"
COLLECTION = "company_knowledge"
EMBED_DIM = 384  # output size of all-MiniLM-L6-v2

# Which chunk labels each role may read. A role missing from this table
# gets nothing: the code fails closed instead of open.
ROLE_ACCESS = {
    "employee": ["employee"],
    "hr_admin": ["employee", "hr_admin"],
}

# Starting value only. Run eval_retrieval.py and adjust it: it must sit above the
# scores of unrelated questions and below the scores of real ones.
MIN_SCORE = 0.30

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

# Changes whenever the policy file changes. The cache stores it with every answer,
# so answers computed from an older version of the policy are never reused.
CORPUS_VERSION = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()[:8]

_indexed = False


def ensure_index():
    """Build the index once, the first time it is needed (not at import time)."""
    global _indexed
    if _indexed:
        return
    chunks = load_and_chunk(POLICY_FILE)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    vectors = encoder.encode([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(id=i, vector=vector.tolist(), payload=chunk)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    _indexed = True


def search(query, role, top_k=1, min_score=MIN_SCORE):
    """Return the best chunks this role is allowed to see, as a list of dicts.

    The role filter runs INSIDE the vector search (pre-filtering), so a forbidden
    chunk is never retrieved, never scored, and never reaches the LLM.
    """
    if role not in ROLE_ACCESS:
        raise ValueError(f"Unknown role: {role!r}")
    ensure_index()
    allowed = Filter(must=[FieldCondition(key="role_required", match=MatchAny(any=ROLE_ACCESS[role]))])
    result = client.query_points(
        collection_name=COLLECTION,
        query=encoder.encode(query).tolist(),
        query_filter=allowed,
        limit=top_k,
        score_threshold=min_score,  # None means no cutoff
    )
    return [
        {"id": p.id, "score": p.score, "text": p.payload["text"], "role_required": p.payload["role_required"]}
        for p in result.points
    ]