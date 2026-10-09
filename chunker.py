def load_and_chunk(file_path: str, chunk_size: int = 150):
    with open(file_path, "r") as f:
        text = f.read()
    
    # Split the document into paragraphs based on double line breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    print(f"--- Loaded {len(paragraphs)} Chunks from {file_path} ---")
    for idx, chunk in enumerate(paragraphs):
        print(f"\nChunk {idx + 1}:\n{chunk}")
    
    return paragraphs

if __name__ == "__main__":
    load_and_chunk("company_policy.txt")