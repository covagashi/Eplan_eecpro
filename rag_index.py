"""
Index all EPLAN EEC Pro markdown docs into ChromaDB for RAG.
Chunks documents, generates embeddings, stores in persistent vector DB.
"""
import os
import re
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = Path("docs_md")
DB_DIR = Path("rag_db")
COLLECTION_NAME = "eecpro_docs"
CHUNK_SIZE = 800  # tokens approx (chars / 4)
CHUNK_OVERLAP = 150


def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown."""
    meta = {}
    body = text
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if match:
        for line in match.group(1).split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                meta[key.strip()] = val.strip().strip('"')
        body = match.group(2)
    return meta, body


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by paragraphs."""
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n{2,}', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > chunk_size * 4:  # ~chars
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Keep overlap from end of previous chunk
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else ''
            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no chunks were created, use the whole text
    if not chunks and text.strip():
        chunks = [text.strip()]

    return chunks


def main():
    print("=" * 60)
    print("EPLAN EEC Pro RAG Indexer")
    print("=" * 60)

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Use ChromaDB default embedding (ONNX all-MiniLM-L6-v2, no PyTorch needed)
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Delete existing collection if exists, recreate
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "EPLAN EEC Pro 2026 documentation"}
    )

    # Process all markdown files
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    all_docs = []
    all_metas = []
    all_ids = []

    for i, md_file in enumerate(md_files):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        meta, body = extract_frontmatter(text)

        if not body.strip():
            continue

        title = meta.get("title", md_file.stem)
        source = meta.get("source", "")
        category = meta.get("category", md_file.parent.name)
        filename = meta.get("file", md_file.stem)

        chunks = chunk_text(body)

        for j, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk{j}"
            # Prepend title context to each chunk for better retrieval
            enriched = f"[{title}] [{category}]\n{chunk}"

            all_docs.append(enriched)
            all_metas.append({
                "title": title,
                "source": source,
                "category": category,
                "file": filename,
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
            all_ids.append(doc_id)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(md_files)} files ({len(all_docs)} chunks)")

    print(f"\nTotal chunks to index: {len(all_docs)}")

    # Insert in batches (ChromaDB limit)
    BATCH = 500
    for start in range(0, len(all_docs), BATCH):
        end = min(start + BATCH, len(all_docs))
        collection.add(
            documents=all_docs[start:end],
            metadatas=all_metas[start:end],
            ids=all_ids[start:end],
        )
        print(f"  Indexed batch {start}-{end}")

    print(f"\nDone! {collection.count()} chunks indexed in {DB_DIR.absolute()}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
