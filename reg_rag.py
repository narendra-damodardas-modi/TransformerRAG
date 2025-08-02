from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TraditionalRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunks: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def _chunk_text(self, text: str) -> List[Tuple[str, str]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i+self.chunk_size])
            chunk_id = f"chunk_{i}_{i+len(chunk_text)}"
            chunks.append((chunk_id, chunk_text))
        return chunks

    def index_corpus(self, corpus: Dict[str, str]):
        for doc_id, text in corpus.items():
            for cid, ctext in self._chunk_text(text):
                self.chunks[cid] = ctext
                embedding = self.model.encode(ctext)
                self.embeddings[cid] = embedding
        print(f"Indexed {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        query_vec = self.model.encode(query)
        scores = []
        for cid, emb in self.embeddings.items():
            score = cosine_similarity([query_vec], [emb])[0][0]
            scores.append((cid, score, self.chunks[cid]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

rag = TraditionalRAG()
docs = dict()
cntr = 1
with open('data.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    li = data.split('\n\n')
    for chunk in li:
        docs[str(cntr)] = chunk
        cntr += 1
rag.index_corpus(docs)
results = rag.retrieve("who is modi and trump?")
for cid, score, content in results:
    print("\n\n")
    print(f"{cid} | score: {score:.4f}\n{content}")
