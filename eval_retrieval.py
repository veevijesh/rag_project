"""Run with: python eval_retrieval.py
Measures retrieval hit rate and helps you choose MIN_SCORE in vector_store.py."""
from vector_store import MIN_SCORE, search

# (question, text that must appear in the top chunk)
IN_SCOPE = [
    ("How many days can I work from home?", "Remote Work Policy"),
    ("What are the core working hours?", "Remote Work Policy"),
    ("Do I need approval to work remotely?", "Remote Work Policy"),
    ("How much can I claim for internet?", "Expense Reimbursement"),
    ("How long do I have to submit receipts?", "Expense Reimbursement"),
    ("Where do I submit expense receipts?", "Expense Reimbursement"),
    ("What bonus do senior engineers get?", "Bonus Band"),
    ("What is the cap on executive bonuses?", "Bonus Band"),
]
OUT_OF_SCOPE = [
    "What is the dress code?",
    "Who won the football match yesterday?",
    "How do I reset my laptop password?",
    "What is the capital of France?",
    "Can I bring my dog to the office?",
    "What are the health insurance benefits?",
]

print("IN-SCOPE questions asked as hr_admin (top-1, no cutoff):")
hits_ok, in_scores = 0, []
for question, marker in IN_SCOPE:
    top = search(question, role="hr_admin", top_k=1, min_score=None)[0]
    ok = marker in top["text"]
    hits_ok += ok
    in_scores.append(top["score"])
    print(f"  {'OK  ' if ok else 'MISS'} score {top['score']:.2f}  {question}")
print(f"Hit rate @1: {hits_ok}/{len(IN_SCOPE)}\n")

print("OUT-OF-SCOPE questions (best score found):")
out_scores = []
for question in OUT_OF_SCOPE:
    top = search(question, role="hr_admin", top_k=1, min_score=None)[0]
    out_scores.append(top["score"])
    print(f"  score {top['score']:.2f}  {question}")

print("\nLEAK CHECK: confidential questions asked as employee")
for question, marker in IN_SCOPE:
    if marker == "Bonus Band":
        top = search(question, role="employee", top_k=1, min_score=None)[0]
        print(f"  {'LEAK!' if marker in top['text'] else 'safe '} {question}")

print(f"\nCurrent MIN_SCORE: {MIN_SCORE}")
print(f"Lowest in-scope score: {min(in_scores):.2f} | Highest out-of-scope score: {max(out_scores):.2f}")
if max(out_scores) < min(in_scores):
    print(f"A cutoff between them, such as {(max(out_scores) + min(in_scores)) / 2:.2f}, separates the two groups.")
else:
    print("The score ranges overlap, so no single cutoff separates them cleanly. Write more test questions.")