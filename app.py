"""Start with: python -m uvicorn app:app --reload"""
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from llm import LLMError
from rag_pipeline import generate_answer
from semantic_cache import cached_answer

app = FastAPI(title="Company Policy RAG API")

# DEMO ONLY. The server decides the role from the API key; the client never sends a role.
# In production, replace this with real authentication (for example a signed JWT from an
# identity provider) and keep secrets out of source control.
API_KEYS = {
    "demo-employee-key": "employee",
    "demo-hr-key": "hr_admin",
}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def role_from_key(api_key):
    role = API_KEYS.get(api_key)
    if role is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return role


@app.post("/ask")
def ask(request: QueryRequest, x_api_key: str | None = Header(default=None)):
    role = role_from_key(x_api_key)
    start = time.perf_counter()
    try:
        result = cached_answer(request.question, role, generate_answer)
    except LLMError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "role": role,
        "question": request.question,
        "answer": result["answer"],
        "cache": result["cache"],
        "sources": result.get("sources", []),
        "latency_ms": latency_ms,
    }