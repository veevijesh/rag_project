from fastapi import FastAPI
from pydantic import BaseModel
from multitenant_rag import client, encoder, collection_name
from qdrant_client.models import Filter, FieldCondition, MatchValue

app = FastAPI(title="Multi-Tenant RAG API")

class QueryRequest(BaseModel):
    question: str
    user_role: str

@app.post("/ask")
def ask_rag(request: QueryRequest):
    search_filter = None
    if request.user_role != "hr_admin":
        search_filter = Filter(
            must=[FieldCondition(key="role_required", match=MatchValue(value="employee"))]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=encoder.encode(request.question).tolist(),
        query_filter=search_filter,
        limit=1
    )

    chunk = results.points[0].payload['text'] if results.points else "Access Denied"
    return {"role": request.user_role, "question": request.question, "answer_chunk": chunk}