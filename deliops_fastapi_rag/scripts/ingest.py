#!/usr/bin/env python3
"""
Script to ingest documents into the vector store.

Usage:
    python -m scripts.ingest --path /path/to/docs

This will read .txt and .md files from the given path, chunk them,
generate embeddings, and store them in the vector index.
"""

import argparse
import os
import sys
from typing import List

import numpy as np

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vectorstore.simple_store import SimpleStore
from app.services.embeddings import embed_texts


# Default index directory
DEFAULT_INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'app', 'vectorstore', 'index')
)


def load_docs(path: str) -> List[dict]:
    """Load text documents from a directory."""
    docs = []
    for root, _, files in os.walk(path):
        for fname in files:
            fp = os.path.join(root, fname)
            if fname.lower().endswith(('.txt', '.md')):
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read().strip()
                    if txt:
                        docs.append({'text': txt[:4000], 'source': fp})
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks


def build_index(docs: List[dict], out_path: str) -> None:
    """Build the vector index from documents."""
    metas = []
    payloads = []

    for d in docs:
        for ch in chunk_text(d['text']):
            metas.append({'text': ch, 'source': d['source']})
            payloads.append(ch)

    if not payloads:
        print("No documents to index")
        return

    print(f"Generating embeddings for {len(payloads)} chunks...")
    vecs = embed_texts(payloads)

    print(f"Building index at {out_path}...")
    store = SimpleStore(out_path)
    store.build(vecs, metas)
    store.save()
    print(f"Successfully indexed {len(metas)} chunks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Ingest documents into the vector store'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Folder containing .txt/.md documents'
    )
    parser.add_argument(
        '--out',
        default=DEFAULT_INDEX_DIR,
        help=f'Output directory for the index (default: {DEFAULT_INDEX_DIR})'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a valid directory")
        sys.exit(1)

    docs = load_docs(args.path)
    print(f"Found {len(docs)} documents")

    if docs:
        build_index(docs, args.out)
    else:
        print("No .txt or .md files found in the specified path")
