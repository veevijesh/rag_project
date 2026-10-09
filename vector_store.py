from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from chunker import load_and_chunk

# 1. Load an embedding model (converts text -> list of 384 numbers)
print("Loading Embedding Model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize in-memory Qdrant database (no cloud setup needed!)
client = QdrantClient(":memory:")

# 3. Create a collection (like a database table for vectors)
collection_name = "company_knowledge"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Chunk document & insert into Qdrant
chunks = load_and_chunk("company_policy.txt")
points = []

for idx, text in enumerate(chunks):
    vector = encoder.encode(text).tolist()
    points.append(PointStruct(id=idx, vector=vector, payload={"text": text}))

client.upsert(collection_name=collection_name, points=points)
print(f"--- Successfully indexed {len(points)} chunks into Qdrant! ---")


def search(query: str, top_k: int = 1):
    """Finds the chunk that best answers the query."""
    query_vector = encoder.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    
    print(f"\nUser Question: '{query}'")
    for res in results.points:
        print(f"Matching Chunk (Score: {res.score:.2f}):\n-> {res.payload['text']}")
    
    return results.points[0].payload['text'] if results.points else None

if __name__ == "__main__":
    search("How much money can I get back for internet?")