import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Initialize Model & Cache Vector Store
encoder = SentenceTransformer("all-MiniLM-L6-v2")
cache_client = QdrantClient(":memory:")
cache_collection = "semantic_cache"

cache_client.create_collection(
    collection_name=cache_collection,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

SIMILARITY_THRESHOLD = 0.75
cache_id_counter = 0


def get_from_cache(query: str):
    """Checks if a semantically similar question was already asked."""
    query_vector = encoder.encode(query).tolist()
    
    results = cache_client.query_points(
        collection_name=cache_collection,
        query=query_vector,
        limit=1
    )
    
    if results.points and results.points[0].score >= SIMILARITY_THRESHOLD:
        match = results.points[0]
        return {
            "cached_answer": match.payload["answer"],
            "matched_question": match.payload["question"],
            "score": match.score
        }
    return None


def save_to_cache(query: str, answer: str):
    """Stores a new question and its generated answer in the cache."""
    global cache_id_counter
    query_vector = encoder.encode(query).tolist()
    
    cache_client.upsert(
        collection_name=cache_collection,
        points=[
            PointStruct(
                id=cache_id_counter,
                vector=query_vector,
                payload={"question": query, "answer": answer}
            )
        ]
    )
    cache_id_counter += 1


def query_with_cache(user_query: str):
    start_time = time.time()
    
    # Check Semantic Cache First
    cached_result = get_from_cache(user_query)
    
    if cached_result:
        latency = (time.time() - start_time) * 1000
        print(f"\n⚡ [CACHE HIT - {latency:.2f}ms]")
        print(f"Query: '{user_query}'")
        print(f"Matched Previous Query: '{cached_result['matched_question']}' (Similarity: {cached_result['score']:.2f})")
        print(f"Answer: {cached_result['cached_answer']}")
    else:
        # Simulate slower primary database search / LLM call
        time.sleep(0.5)  # Simulates 500ms delay for actual retrieval
        generated_answer = "Employees can claim up to $50 per month for home internet expenses."
        
        # Save to cache for future users
        save_to_cache(user_query, generated_answer)
        
        latency = (time.time() - start_time) * 1000
        print(f"\n🐢 [CACHE MISS - {latency:.2f}ms]")
        print(f"Query: '{user_query}'")
        print(f"Answer: {generated_answer}")


if __name__ == "__main__":
    # Query 1: First time asking (Cache Miss -> DB search)
    query_with_cache("How much can I claim for home internet?")
    
    # Query 2: Rephrased question asking the exact same thing (Cache Hit!)
    query_with_cache("What is the internet reimbursement limit?")