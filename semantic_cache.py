"""A role-aware semantic cache: reuse an earlier answer when a new question means the same thing."""
import re
import uuid

from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from vector_store import CORPUS_VERSION, EMBED_DIM, ROLE_ACCESS, client, encoder

CACHE_COLLECTION = "semantic_cache"

# Chosen from tune_cache_threshold.py on 12 question pairs: the best paraphrase similarities
# were 0.88, 0.79, 0.77, and the highest near-miss (ignoring the number check below) was 0.69.
# 0.75 sits above that with a margin. Raising it means fewer hits, lowering it risks wrong answers.
CACHE_THRESHOLD = 0.75

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_match(a, b):
    """True if both questions contain the same numbers.

    Embedding similarity barely notices numbers: "$50" vs "$500" scored 0.92 in testing,
    higher than every genuine paraphrase. Without this check the cache could answer a
    question about $500 with the stored answer for $50.
    """
    return set(_NUMBER.findall(a)) == set(_NUMBER.findall(b))


def _create_collection():
    client.create_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


_create_collection()


def clear_cache():
    client.delete_collection(CACHE_COLLECTION)
    _create_collection()


def _cache_filter(role):
    """An entry is usable only if it was made for THIS role and THIS version of the policy."""
    return Filter(must=[
        FieldCondition(key="role", match=MatchValue(value=role)),
        FieldCondition(key="corpus_version", match=MatchValue(value=CORPUS_VERSION)),
    ])


def get_from_cache(question, role):
    results = client.query_points(
        collection_name=CACHE_COLLECTION,
        query=encoder.encode(question.strip()).tolist(),
        query_filter=_cache_filter(role),
        limit=5,
        score_threshold=CACHE_THRESHOLD,
    )
    for match in results.points:  # best match first; take the first one whose numbers agree
        if numbers_match(question, match.payload["question"]):
            return {
                "answer": match.payload["answer"],
                "sources": match.payload["sources"],
                "matched_question": match.payload["question"],
                "score": round(match.score, 3),
            }
    return None


def save_to_cache(question, answer, sources, role):
    client.upsert(
        collection_name=CACHE_COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=encoder.encode(question.strip()).tolist(),
            payload={
                "question": question, "answer": answer, "sources": sources,
                "role": role, "corpus_version": CORPUS_VERSION,
            },
        )],
    )


def cached_answer(question, role, generate_fn):
    """Return the cached answer if one is close enough, otherwise call generate_fn and cache it.

    Only real, successful answers are cached: not "nothing found" and not
    fallback text produced when no LLM is configured.
    """
    if role not in ROLE_ACCESS:
        raise ValueError(f"Unknown role: {role!r}")
    hit = get_from_cache(question, role)
    if hit:
        return {**hit, "found": True, "generated": True, "cache": "HIT"}
    result = generate_fn(question, role)
    if result["found"] and result["generated"]:
        save_to_cache(question, result["answer"], result["sources"], role)
    return {**result, "cache": "MISS"}