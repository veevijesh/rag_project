from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# 1. Initialize Model & Qdrant
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")
collection_name = "multitenant_knowledge"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 2. Add Documents with Role Permissions (Payload Filtering)
documents = [
    {
        "text": "Employees are allowed to work remotely up to 2 days per week with manager approval.",
        "role": "employee"  # Publicly visible to all employees
    },
    {
        "text": "Employees can claim up to $50 per month for home internet expenses.",
        "role": "employee"  # Publicly visible to all employees
    },
    {
        "text": "Senior Engineers receive a $15,000 performance bonus. Executive leadership bonuses are capped at 30%.",
        "role": "hr_admin"  # CONFIDENTIAL: HR and Admins only!
    }
]

points = []
for idx, doc in enumerate(documents):
    vector = encoder.encode(doc["text"]).tolist()
    points.append(PointStruct(
        id=idx, 
        vector=vector, 
        payload={"text": doc["text"], "role_required": doc["role"]}
    ))

client.upsert(collection_name=collection_name, points=points)
print("--- Multi-tenant Knowledge Base Indexed with Role Security! ---\n")


def secure_search(query: str, user_role: str):
    """Searches the database, filtering out results the user isn't allowed to see."""
    query_vector = encoder.encode(query).tolist()
    
    # SECURITY FILTER:
    # If user is 'employee', they can ONLY see 'employee' documents.
    # If user is 'hr_admin', they can see 'hr_admin' AND 'employee' documents.
    
    search_filter = None
    if user_role != "hr_admin":
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="role_required",
                    match=MatchValue(value="employee")
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=search_filter,
        limit=1
    )

    print(f"User Role: [{user_role.upper()}] | Question: '{query}'")
    if results.points:
        print(f"-> Allowed Match: {results.points[0].payload['text']}\n")
    else:
        print("-> [ACCESS DENIED / NO MATCH]: You do not have permission to view this information.\n")


if __name__ == "__main__":
    # TEST 1: Regular Employee asks about bonuses
    secure_search("What is the bonus policy for engineers?", user_role="employee")
    
    # TEST 2: HR Admin asks the exact same question
    secure_search("What is the bonus policy for engineers?", user_role="hr_admin")