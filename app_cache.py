from fastapi import FastAPI
from pydantic import BaseModel
from semantic_cache import query_with_cache, get_from_cache, save_to_cache
import time

app = FastAPI(title="Semantic Caching RAG API")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask_cached")
def ask_cached(request: QueryRequest):
    start_time = time.time()
    
    # 1. Check Cache
    cached_result = get_from_cache(request.question)
    
    if cached_result:
        latency = (time.time() - start_time) * 1000
        return {
            "status": "CACHE_HIT",
            "latency_ms": round(latency, 2),
            "question": request.question,
            "matched_question": cached_result["matched_question"],
            "similarity_score": round(cached_result["score"], 2),
            "answer": cached_result["cached_answer"]
        }
    
    # 2. Cache Miss (Simulated DB/LLM lookup)
    time.sleep(0.4)
    answer = "Employees can claim up to $50 per month for home internet expenses."
    save_to_cache(request.question, answer)
    
    latency = (time.time() - start_time) * 1000
    return {
        "status": "CACHE_MISS",
        "latency_ms": round(latency, 2),
        "question": request.question,
        "answer": answer
    }