"""Retrieve the allowed chunks, then ask the LLM to answer from them."""
from llm import LLMError, LLMNotConfigured, generate
from vector_store import search

PROMPT = """You are a company policy assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know. Be concise.

Context:
{context}

Question: {question}
Answer:"""

NOT_FOUND = "I couldn't find relevant information in the policies available to you."


def generate_answer(question, role, top_k=2):
    """Return {"answer", "found", "generated", "sources"}.

    found:     retrieval returned at least one chunk above the similarity cutoff
    generated: the answer was written by the LLM (False if no LLM is configured)
    """
    hits = search(question, role=role, top_k=top_k)
    if not hits:
        return {"answer": NOT_FOUND, "found": False, "generated": False, "sources": []}

    sources = [{"id": h["id"], "score": round(h["score"], 3)} for h in hits]
    context = "\n\n".join(h["text"] for h in hits)
    try:
        answer = generate(PROMPT.format(context=context, question=question))
        generated = True
    except LLMNotConfigured:
        answer = "(LLM not configured; showing the closest policy text) " + hits[0]["text"]
        generated = False
    return {"answer": answer, "found": True, "generated": generated, "sources": sources}


if __name__ == "__main__":
    try:
        for role in ("employee", "hr_admin"):
            result = generate_answer("What bonus do senior engineers get?", role)
            print(f"\n[{role}] {result['answer']}\n sources: {result['sources']}")
        print("\n[employee]", generate_answer("What is the policy on working from home?", "employee")["answer"])
    except LLMError as error:
        print(f"\nThe LLM call failed: {error}")
        print("This is often temporary. Wait a minute and run it again, or set a different LLM_MODEL.")