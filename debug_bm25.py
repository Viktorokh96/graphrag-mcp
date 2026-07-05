from src.bm25_index import BM25Index
import re

index = BM25Index()
index.add_document("doc1", "python programming language")

# Test tokenizer
text = "python programming language"
tokens = re.findall(r'\b\w+\b', text)
print("tokens:", tokens)

# Test BM25 directly
from rank_bm25 import BM25Okapi
# index._documents — это список списков токенов
tokenized = index._documents
print("tokenized:", tokenized)
bm25 = BM25Okapi(tokenized)
query_tokens = re.findall(r'\b\w+\b', "python")
print("query_tokens:", query_tokens)
scores = bm25.get_scores(query_tokens)
print("scores:", scores)

results = index.search("python", k=5)
print("results:", results)