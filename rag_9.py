import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FlowRAGConfig:
    model_name: str = "distilbert-base-uncased"
    max_seq_length: int = 512
    attention_layer: int = -1
    attention_threshold: float = 0.1
    expansion_iters: int = 3
    max_results: int = 10
    chunk_size: int = 200  # tokens per paragraph chunk
    device: str = "auto"

@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    reasoning: str

class TokenGraph:
    def __init__(self):
        self.adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.token_blocks: Dict[int, Set[str]] = defaultdict(set)
        self.block_tokens: Dict[str, Set[int]] = {}

    def add_block(self, block_id: str, tokens: Set[int]):
        self.block_tokens[block_id] = tokens
        for t in tokens:
            self.token_blocks[t].add(block_id)

    def add_edge(self, i: int, j: int, w: float, block_id: str):
        self.adj[i].append((j, w))
        self.token_blocks[i].add(block_id)
        self.token_blocks[j].add(block_id)

    def neighbors(self, token: int, thresh: float) -> List[Tuple[int, float]]:
        return [(j, w) for j, w in self.adj.get(token, []) if w >= thresh]

    def expand(self, tokens: Set[int], thresh: float, iters: int) -> Set[int]:
        expanded = set(tokens)
        for _ in range(iters):
            new = set()
            for t in list(expanded):
                for nb, w in self.neighbors(t, thresh):
                    new.add(nb)
            expanded |= new
        return expanded

    def blocks_for_tokens(self, tokens: Set[int]) -> Set[str]:
        blocks = set()
        for t in tokens:
            blocks |= self.token_blocks.get(t, set())
        return blocks

class FlowRAGChunked:
    def __init__(self, config: FlowRAGConfig):
        self.config = config
        self._setup_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.graph = TokenGraph()
        self.chunks: Dict[str, str] = {}

    def _setup_device(self):
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")

    def _chunk_text(self, text: str) -> List[Tuple[str, str]]:
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_attention_mask=False)
        tokens = inputs.input_ids
        chunks = []
        current, start = [], 0
        for i, tok in enumerate(tokens):
            current.append(tok)
            if len(current) >= self.config.chunk_size:
                chunk_tokens = current.copy()
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk_id = f"chunk_{start}_{i}"
                chunks.append((chunk_id, chunk_text))
                current, start = [], i+1
        if current:
            text_chunk = self.tokenizer.decode(current, skip_special_tokens=True)
            chunk_id = f"chunk_{start}_{len(tokens)-1}"
            chunks.append((chunk_id, text_chunk))
        return chunks

    def index_corpus(self, corpus: Dict[str, str]):
        for doc_id, text in corpus.items():
            for cid, ctext in self._chunk_text(text):
                self.chunks[cid] = ctext

        logger.info(f"Total chunks: {len(self.chunks)}")
        for cid, ctext in self.chunks.items():
            enc = self.tokenizer(ctext, truncation=True, padding=True,
                                 max_length=self.config.max_seq_length, return_tensors='pt')
            input_ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids, attention_mask=mask, output_attentions=True)
            attn = out.attentions[self.config.attention_layer].mean(1).squeeze(0)
            seq_len = mask.sum().item()
            toks = input_ids.squeeze(0)[:seq_len].cpu().numpy()
            self.graph.add_block(cid, set(toks))
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j and attn[i,j] > self.config.attention_threshold:
                        self.graph.add_edge(int(toks[i]), int(toks[j]), float(attn[i,j]), cid)
        logger.info("Token graph built.")

    def retrieve(self, query: str) -> List[RetrievalResult]:
        enc = self.tokenizer(query, truncation=True, return_tensors='pt').to(self.device)
        qids = set(enc.input_ids.squeeze(0).cpu().numpy())
        expanded = self.graph.expand(qids, self.config.attention_threshold, self.config.expansion_iters)
        blocks = self.graph.blocks_for_tokens(expanded)
        scored = []
        for bid in blocks:
            btoks = self.graph.block_tokens.get(bid, set())
            common = btoks & expanded
            score = len(common) / len(btoks) if btoks else 0
            scored.append((bid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for bid, score in scored[: self.config.max_results]:
            results.append(RetrievalResult(
                chunk_id=bid,
                content=self.chunks[bid],
                score=score,
                reasoning=f"{len(btoks & expanded)} of {len(btoks)} tokens matched"
            ))
        return results

config = FlowRAGConfig()
rag = FlowRAGChunked(config)
docs = dict()
cntr=1
with open('data.txt','r',encoding='UTF-8') as f:
    data = f.read()
    li = data.split('\n\n')
    for chunk in li:
        docs[str(cntr)]=chunk
rag.index_corpus(docs)
res = rag.retrieve('who is modi and trump?')
for r in res:
    print('\n\n\n')
    print(r.chunk_id, r.score, r.content)