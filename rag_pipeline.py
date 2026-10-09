from vector_store import search

def generate_answer(user_question: str):
    # Step 1: RETRIEVE relevant knowledge
    retrieved_chunk = search(user_question, top_k=1)
    
    if not retrieved_chunk:
        return "I couldn't find any relevant policy information."
    
    # Step 2: GENERATE answer using context
    # (Simulating LLM synthesis prompt for baseline)
    response = (
        f"🤖 AI Answer:\n"
        f"Based on company policy: {retrieved_chunk}\n\n"
        f"Summary: You can claim up to $50/month for home internet, "
        f"provided you submit receipts within 30 days."
    )
    return response

if __name__ == "__main__":
    question = "What is the policy on working from home?"
    print(f"\n--- Testing Full RAG Pipeline ---")
    answer = generate_answer(question)
    print("\n" + answer)