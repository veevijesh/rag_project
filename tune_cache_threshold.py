"""Run with: python tune_cache_threshold.py
Sweeps the cache threshold over paraphrase pairs (should hit) and near-miss pairs (must not hit).
A near-miss only counts as a false hit if the cache's number check would also let it through."""
from sentence_transformers import util

from semantic_cache import numbers_match
from vector_store import encoder

PARAPHRASES = [
    ("How much can I claim for home internet?", "What is the internet reimbursement limit?"),
    ("How many days can I work remotely?", "How many remote work days are allowed per week?"),
    ("What are the core working hours?", "Which hours do I need to be online?"),
    ("When do I need to submit my expense receipts?", "What is the deadline for submitting receipts?"),
    ("What bonus do senior engineers get?", "How much is the performance bonus for senior engineers?"),
    ("Do I need approval to work from home?", "Does remote work require manager approval?"),
]
NEAR_MISSES = [
    ("How much can I claim for home internet?", "How much can I claim for travel?"),
    ("How many days can I work remotely?", "How many days of annual leave do I get?"),
    ("What are the core working hours?", "What are the office opening hours?"),
    ("What bonus do senior engineers get?", "What bonus do executives get?"),
    ("When do I need to submit receipts?", "When do I get reimbursed?"),
    ("Can I claim $50 per month for internet?", "Can I claim $500 per month for internet?"),
]


def similarity(a, b):
    return float(util.cos_sim(encoder.encode(a), encoder.encode(b)))


para_scores = [similarity(a, b) for a, b in PARAPHRASES]
miss_scores = [similarity(a, b) for a, b in NEAR_MISSES]
# A near-miss that the number check blocks can never be a false hit, whatever its score.
miss_blocked = [not numbers_match(a, b) for a, b in NEAR_MISSES]

print("PARAPHRASE pairs (should hit):")
for (a, b), s in zip(PARAPHRASES, para_scores):
    print(f"  {s:.2f}  {a}  <->  {b}")
print("\nNEAR-MISS pairs (must NOT hit):")
for (a, b), s, blocked in zip(NEAR_MISSES, miss_scores, miss_blocked):
    note = "  [blocked by number check]" if blocked else ""
    print(f"  {s:.2f}  {a}  <->  {b}{note}")

print("\nthreshold | paraphrases hit | false hits")
best = None
for i in range(60, 98, 2):
    t = i / 100
    hits = sum(s >= t for s in para_scores)
    false_hits = sum(s >= t and not blocked for s, blocked in zip(miss_scores, miss_blocked))
    print(f"  {t:.2f}    | {hits}/{len(para_scores)}            | {false_hits}")
    if false_hits == 0 and hits > 0 and best is None:
        best = t

if best is None:
    print("\nNo threshold both catches paraphrases and avoids false hits. Add more pairs or change the model.")
else:
    print(f"\nLowest threshold with zero false hits AND at least one paraphrase hit: {best:.2f}")
    print(f"Add a safety margin: CACHE_THRESHOLD around {best + 0.05:.2f} in semantic_cache.py.")
    print("A wrong cached answer is worse than a missed cache hit, so err on the high side.")