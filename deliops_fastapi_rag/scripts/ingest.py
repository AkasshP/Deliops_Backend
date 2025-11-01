
import argparse, os, glob
from typing import List
from app.vectorstore.faiss_store import FaissStore
from app.services.embeddings import embed_texts

def load_docs(path: str) -> List[dict]:
    docs = []
    for root, _, files in os.walk(path):
        for fname in files:
            fp = os.path.join(root, fname)
            if fname.lower().endswith(('.txt','.md')):
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read().strip()
                    if txt:
                        docs.append({'text': txt[:4000], 'source': fp})
    return docs

def chunk_text(text: str, chunk_size=800, overlap=120):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def build_index(docs: List[dict], out_path: str):
    metas, payloads = [], []
    for d in docs:
        for ch in chunk_text(d['text']):
            metas.append({'text': ch, 'source': d['source']})
            payloads.append(ch)
    if not payloads:
        print("No docs to index")
        return
    import numpy as np
    vecs = embed_texts(payloads)
    store = FaissStore(out_path)
    store.build(vecs, metas)
    store.save()
    print(f"Indexed {len(metas)} chunks -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, help='Folder containing .txt/.md documents')
    ap.add_argument('--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','app','vectorstore','index.faiss')))
    args = ap.parse_args()
    docs = load_docs(args.path)
    build_index(docs, args.out)
