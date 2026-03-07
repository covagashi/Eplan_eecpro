"""
EPLAN EEC Pro RAG Indexer v2

Mejoras sobre v1:
1. Chunking jerarquico por headers markdown (no por parrafos)
2. Embeddings mejorados (bge-base-en-v1.5 via sentence-transformers)
3. Metadata enriquecida (header_path, category, parent_id)
4. Parent-child chunks (child para buscar, parent para contexto)
5. Exporta corpus BM25 para hybrid search en query time
"""
import os
import re
import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DOCS_DIR = Path("docs_md")
DB_DIR = Path("rag_db_v2")
COLLECTION_NAME = "eecpro_v2"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
MAX_CHILD_CHARS = 2500
MIN_CHUNK_CHARS = 80


def extract_frontmatter(text: str) -> tuple[dict, str]:
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


def parse_sections(body: str) -> list[dict]:
    """Split markdown into sections by H1/H2/H3 headers.
    Returns list of {level, header, content, header_path}."""
    lines = body.split('\n')
    sections = []
    headers = {1: '', 2: '', 3: ''}
    current_lines = []
    current_level = 0

    def flush():
        text = '\n'.join(current_lines).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            path_parts = [headers[l] for l in (1, 2, 3) if headers[l]]
            sections.append({
                'level': current_level,
                'header': headers.get(current_level, ''),
                'content': text,
                'header_path': ' > '.join(path_parts),
            })

    for line in lines:
        m = re.match(r'^(#{1,3})\s+(.+)', line)
        if m:
            flush()
            current_lines = [line]
            level = len(m.group(1))
            headers[level] = m.group(2).strip()
            for l in range(level + 1, 4):
                headers[l] = ''
            current_level = level
        else:
            current_lines.append(line)

    flush()
    return sections


def split_long_section(section: dict, max_chars: int = MAX_CHILD_CHARS) -> list[dict]:
    """If a section is too long, split by paragraphs."""
    content = section['content']
    if len(content) <= max_chars:
        return [section]

    paragraphs = re.split(r'\n{2,}', content)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chars and len(current) >= MIN_CHUNK_CHARS:
            chunks.append({
                **section,
                'content': current.strip(),
            })
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip() and len(current.strip()) >= MIN_CHUNK_CHARS:
        chunks.append({**section, 'content': current.strip()})

    return chunks if chunks else [section]


def main():
    print("=" * 60)
    print("EPLAN EEC Pro RAG Indexer v2")
    print("=" * 60)

    DB_DIR.mkdir(exist_ok=True)

    # Embedding model: bge-base-en-v1.5 (384 -> 768 dim, much better for technical text)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        normalize_embeddings=True,
    )

    # ChromaDB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "EPLAN EEC Pro 2026 v2 - hierarchical chunks"}
    )

    md_files = sorted(DOCS_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files\n")

    # Storage for parent map and BM25 corpus
    parent_map = {}   # parent_id -> full page text
    bm25_corpus = []  # [{id, text, category, title}]

    all_docs = []
    all_metas = []
    all_ids = []

    for i, md_file in enumerate(md_files):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        meta, body = extract_frontmatter(text)

        if not body.strip() or len(body.strip()) < MIN_CHUNK_CHARS:
            continue

        title = meta.get("title", md_file.stem)
        source = meta.get("source", "")
        category = meta.get("category", md_file.parent.name)
        filename = meta.get("file", md_file.stem)
        parent_id = f"parent_{filename}"

        # Store full page as parent
        parent_map[parent_id] = {
            "title": title,
            "category": category,
            "source": source,
            "content": body.strip(),
        }

        # Parse into hierarchical sections
        sections = parse_sections(body)
        if not sections:
            # Short page: use whole body as single chunk
            sections = [{'level': 0, 'header': title, 'content': body.strip(), 'header_path': title}]

        # Split long sections into sub-chunks
        child_chunks = []
        for sec in sections:
            child_chunks.extend(split_long_section(sec))

        # Index each child chunk
        for j, chunk in enumerate(child_chunks):
            child_id = f"{filename}_c{j}"
            header_path = chunk['header_path'] or title

            # Prepend context for better retrieval
            enriched = f"[{category}] [{header_path}]\n{chunk['content']}"

            all_docs.append(enriched)
            all_metas.append({
                "title": title,
                "source": source,
                "category": category,
                "file": filename,
                "parent_id": parent_id,
                "header_path": header_path,
                "header": chunk.get('header', ''),
                "chunk_index": j,
                "total_chunks": len(child_chunks),
                "level": chunk['level'],
            })
            all_ids.append(child_id)

            # BM25 corpus entry
            bm25_corpus.append({
                "id": child_id,
                "text": chunk['content'].lower(),
                "category": category,
                "title": title,
            })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(md_files)} files ({len(all_docs)} chunks)")

    print(f"\nTotal child chunks: {len(all_docs)}")
    print(f"Total parent pages: {len(parent_map)}")

    # Insert in batches
    BATCH = 200
    for start in range(0, len(all_docs), BATCH):
        end = min(start + BATCH, len(all_docs))
        collection.add(
            documents=all_docs[start:end],
            metadatas=all_metas[start:end],
            ids=all_ids[start:end],
        )
        print(f"  Indexed batch {start}-{end}")

    # Save parent map
    parent_path = DB_DIR / "parent_map.json"
    with open(parent_path, 'w', encoding='utf-8') as f:
        json.dump(parent_map, f, ensure_ascii=False)
    print(f"  Parent map saved: {parent_path}")

    # Save BM25 corpus
    bm25_path = DB_DIR / "bm25_corpus.json"
    with open(bm25_path, 'w', encoding='utf-8') as f:
        json.dump(bm25_corpus, f, ensure_ascii=False)
    print(f"  BM25 corpus saved: {bm25_path}")

    print(f"\nDone! {collection.count()} chunks indexed in {DB_DIR.absolute()}")


if __name__ == "__main__":
    main()
