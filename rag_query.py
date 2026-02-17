"""
Query the EPLAN EEC Pro RAG system.
Usage: python rag_query.py "your question here" [--top N]
"""
import sys
import json
import argparse
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = Path("rag_db")
COLLECTION_NAME = "eecpro_docs"


def query_rag(question: str, top_k: int = 5, category: str = None) -> list[dict]:
    """Query the RAG and return relevant chunks."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    where_filter = {"category": category} if category else None

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "distance": round(results["distances"][0][i], 4),
            "title": results["metadatas"][0][i]["title"],
            "source": results["metadatas"][0][i]["source"],
            "category": results["metadatas"][0][i]["category"],
            "file": results["metadatas"][0][i]["file"],
            "chunk": results["metadatas"][0][i]["chunk_index"],
            "content": results["documents"][0][i],
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Query EPLAN EEC Pro documentation RAG")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = query_rag(args.question, args.top, args.category)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Query: {args.question}")
        print(f"Results: {len(results)}")
        print(f"{'='*60}\n")

        for i, r in enumerate(results, 1):
            print(f"--- Result {i} (distance: {r['distance']}) ---")
            print(f"Title:    {r['title']}")
            print(f"File:     {r['file']}")
            print(f"Category: {r['category']}")
            print(f"Source:   {r['source']}")
            print(f"Chunk:    {r['chunk']}")
            print(f"\n{r['content'][:600]}...")
            print()


if __name__ == "__main__":
    main()
