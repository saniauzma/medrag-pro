import pickle

with open("data/indexes/bm25_corpus.pkl", "rb") as f:
    data = pickle.load(f)
for d in data.corpus:
    print(d.content)