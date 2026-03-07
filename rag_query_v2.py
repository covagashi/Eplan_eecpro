"""
EPLAN EEC Pro RAG Query v2

Mejoras sobre v1:
3. Hybrid search: BM25 (keyword) + vector (semantico) con Reciprocal Rank Fusion
4. Metadata enrichment: filtra por categoria, muestra header_path
5. Query expansion: sinonimos tecnicos EPLAN
6. Reranking con cross-encoder (opcional, mejora precision)
7. Parent-child: devuelve el contexto completo de la pagina padre

Usage:
  python rag_query_v2.py "how to configure REST API" --top 5
  python rag_query_v2.py "GenerateEcadCodeCommand" --category refcommands
  python rag_query_v2.py "formula if then else" --no-parent
"""
import sys
import json
import argparse
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = Path("rag_db_v2")
COLLECTION_NAME = "eecpro_v2"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- Punto 5: Query expansion ---
# Sinonimos tecnicos para EPLAN EEC Pro
SYNONYMS = {
    "formula": ["expression", "calculation", "computed", "function"],
    "terminal": ["connection point", "addressing", "connector"],
    "plc": ["programmable logic controller", "io", "fieldbus"],
    "ecad": ["electrical cad", "schematic", "wiring", "circuit"],
    "graph2d": ["2d graphic", "diagram", "drawing", "visualization"],
    "macro": ["template", "reusable", "symbol"],
    "bom": ["bill of materials", "parts list", "material list"],
    "csv": ["export", "import", "comma separated", "data exchange"],
    "xml": ["export", "import", "data exchange", "markup"],
    "job": ["jobserver", "batch", "automation", "scheduled task"],
    "api": ["rest", "interface", "endpoint", "webservice"],
    "script": ["scripting", "eecscripting", "automation", "code"],
    "panel": ["propanel", "cabinet", "enclosure", "layout"],
    "sap": ["erp", "integration", "material management"],
    "form": ["formui", "dialog", "user interface", "ui"],
    "command": ["action", "execute", "run", "operation"],
    "parameter": ["property", "attribute", "setting", "value"],
    "project": ["model", "workspace", "configuration"],
    "report": ["output", "document", "listing", "printout"],
    "backup": ["export", "save", "restore", "archive"],
    "addressing": ["terminal", "numbering", "io addressing"],
    "regex": ["regular expression", "pattern", "match", "text pattern"],
}


def expand_query(query: str) -> str:
    """Expand query with EPLAN-specific synonyms."""
    words = query.lower().split()
    expansions = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if clean in SYNONYMS:
            expansions.extend(SYNONYMS[clean][:2])  # max 2 synonyms per word
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# --- Punto 3: Hybrid search (BM25 + vector) ---

def load_bm25():
    """Load BM25 index from saved corpus."""
    bm25_path = DB_DIR / "bm25_corpus.json"
    with open(bm25_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    texts = [doc["text"].split() for doc in corpus]
    ids = [doc["id"] for doc in corpus]
    categories = {doc["id"]: doc["category"] for doc in corpus}
    bm25 = BM25Okapi(texts)
    return bm25, ids, categories, texts


def bm25_search(query: str, bm25, ids, categories, top_k=20, category=None):
    """BM25 keyword search. Returns list of (id, score)."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    results = list(zip(ids, scores))

    if category:
        results = [(id_, s) for id_, s in results if categories.get(id_) == category]

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def vector_search(query: str, collection, top_k=20, category=None):
    """ChromaDB vector search. Returns list of (id, score, metadata, doc)."""
    where_filter = {"category": category} if category else None
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for i in range(len(results["ids"][0])):
        output.append((
            results["ids"][0][i],
            results["distances"][0][i],
            results["metadatas"][0][i],
            results["documents"][0][i],
        ))
    return output


def reciprocal_rank_fusion(ranked_lists: list[list[tuple]], k=60) -> list[str]:
    """Merge multiple ranked lists using RRF. Returns ordered list of IDs."""
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            doc_id = item[0]  # first element is always ID
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids


# --- Punto 6: Cross-encoder reranking ---

_reranker = None

def get_reranker():
    """Lazy-load cross-encoder for reranking."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("  [Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2]")
        except Exception as e:
            print(f"  [Reranker not available: {e}]")
            _reranker = False
    return _reranker if _reranker else None


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates with cross-encoder. Falls back to original order."""
    reranker = get_reranker()
    if not reranker or not candidates:
        return candidates[:top_k]

    pairs = [(query, c["content"][:512]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# --- Punto 7: Parent-child retrieval ---

def load_parent_map():
    parent_path = DB_DIR / "parent_map.json"
    with open(parent_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# --- Main query pipeline ---

def query_rag(question: str, top_k: int = 5, category: str = None,
              use_parent: bool = True, use_rerank: bool = True) -> list[dict]:
    """Full RAG query pipeline: expand -> hybrid search -> RRF -> rerank -> parent context."""

    # 1. Load resources
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, normalize_embeddings=True,
    )
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    bm25, bm25_ids, bm25_cats, _ = load_bm25()
    parent_map = load_parent_map() if use_parent else {}

    # 2. Expand query
    expanded = expand_query(question)

    # 3. Hybrid search
    retrieve_k = max(top_k * 4, 20)  # retrieve more for reranking
    vec_results = vector_search(expanded, collection, retrieve_k, category)
    bm25_results = bm25_search(expanded, bm25, bm25_ids, bm25_cats, retrieve_k, category)

    # 4. Reciprocal Rank Fusion
    fused_ids = reciprocal_rank_fusion([vec_results, bm25_results])

    # Build lookup from vector results (they have metadata)
    vec_lookup = {r[0]: {"id": r[0], "distance": r[1], "meta": r[2], "content": r[3]}
                  for r in vec_results}

    # For BM25-only hits, fetch from collection
    missing_ids = [id_ for id_ in fused_ids[:retrieve_k] if id_ not in vec_lookup]
    if missing_ids:
        try:
            fetched = collection.get(ids=missing_ids, include=["documents", "metadatas"])
            for i, id_ in enumerate(fetched["ids"]):
                vec_lookup[id_] = {
                    "id": id_,
                    "distance": -1,  # unknown
                    "meta": fetched["metadatas"][i],
                    "content": fetched["documents"][i],
                }
        except Exception:
            pass

    # Build candidate list in RRF order
    candidates = []
    for id_ in fused_ids:
        if id_ in vec_lookup:
            r = vec_lookup[id_]
            candidates.append({
                "id": id_,
                "distance": r["distance"],
                "title": r["meta"].get("title", ""),
                "source": r["meta"].get("source", ""),
                "category": r["meta"].get("category", ""),
                "file": r["meta"].get("file", ""),
                "header_path": r["meta"].get("header_path", ""),
                "parent_id": r["meta"].get("parent_id", ""),
                "chunk_index": r["meta"].get("chunk_index", 0),
                "content": r["content"],
            })

    # 5. Rerank with cross-encoder
    if use_rerank:
        candidates = rerank(question, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    # 6. Attach parent context
    if use_parent:
        for c in candidates:
            pid = c.get("parent_id", "")
            if pid in parent_map:
                c["parent_content"] = parent_map[pid]["content"]
            else:
                c["parent_content"] = ""

    return candidates


def main():
    parser = argparse.ArgumentParser(description="EPLAN EEC Pro RAG Query v2")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--no-parent", action="store_true", help="Don't show parent context")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = query_rag(
        args.question, args.top, args.category,
        use_parent=not args.no_parent,
        use_rerank=not args.no_rerank,
    )

    if args.json:
        # Don't include parent_content in JSON (too big)
        for r in results:
            r.pop("parent_content", None)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Query: {args.question}")
        expanded = expand_query(args.question)
        if expanded != args.question:
            print(f"Expanded: {expanded}")
        print(f"Results: {len(results)}")
        print(f"{'='*60}\n")

        for i, r in enumerate(results, 1):
            score_info = f"dist: {r['distance']:.4f}" if r['distance'] >= 0 else "BM25 only"
            if "rerank_score" in r:
                score_info += f", rerank: {r['rerank_score']:.4f}"
            print(f"--- Result {i} ({score_info}) ---")
            print(f"Title:    {r['title']}")
            print(f"Path:     {r['header_path']}")
            print(f"Category: {r['category']}")
            print(f"Source:   {r['source']}")
            print(f"\n{r['content'][:800]}")
            if r.get("parent_content") and not args.no_parent:
                print(f"\n  [Parent page: {len(r['parent_content'])} chars available]")
            print()


if __name__ == "__main__":
    main()
