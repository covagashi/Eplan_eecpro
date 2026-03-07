# EPLAN EEC Pro 2026 - Documentation RAG

Scraper + RAG (Retrieval Augmented Generation) de la documentacion oficial de **EPLAN Engineering Configuration Pro 2026**.

## Contenido

- `docs_md/` - 1648 paginas de documentacion convertidas a Markdown, organizadas en 36 categorias
- `rag_db/` - Base de datos vectorial ChromaDB v1 (1967 chunks, embedding MiniLM-L6)
- `rag_db_v2/` - Base de datos vectorial v2 (3043 chunks, embedding bge-base-en-v1.5)
- `scrape_eplan.py` - Script de scraping de la documentacion
- `rag_index.py` - Indexador v1 (basico, chunking por parrafos)
- `rag_index_v2.py` - Indexador v2 (chunking jerarquico, parent-child, metadata enriquecida)
- `rag_query.py` - Buscador v1 (solo vectorial)
- `rag_query_v2.py` - Buscador v2 (hybrid BM25+vector, reranking, query expansion)

## Uso rapido (v2 recomendado)

```bash
# Instalar dependencias
pip install requests beautifulsoup4 html2text chromadb sentence-transformers rank_bm25

# Re-indexar (solo si se modifican los docs)
python rag_index_v2.py

# Buscar en la documentacion
python rag_query_v2.py "how to configure Job Server REST API" --top 5
python rag_query_v2.py "GenerateEcadCodeCommand" --top 3
python rag_query_v2.py "formula if then else" --category refformulas

# Opciones
python rag_query_v2.py "terminal addressing" --no-rerank    # sin cross-encoder (mas rapido)
python rag_query_v2.py "macro template" --no-parent          # sin contexto padre
python rag_query_v2.py "csv export" --json                   # salida JSON
```

## v1 vs v2: que cambio

| Aspecto | v1 | v2 |
|---|---|---|
| Chunking | Por parrafos (800 chars) | Jerarquico por headers H1/H2/H3 |
| Chunks | 1.967 | 3.043 child + 1.648 parent |
| Embedding | all-MiniLM-L6-v2 (ONNX) | bge-base-en-v1.5 (sentence-transformers) |
| Busqueda | Solo vectorial | Hybrid: BM25 + vectorial + RRF |
| Reranking | No | Cross-encoder ms-marco-MiniLM |
| Query expansion | No | Sinonimos tecnicos EPLAN |
| Contexto | Solo el chunk | Chunk + pagina padre completa |
| Metadata | Basica (title, category) | Rica (header_path, parent_id, level) |

## Categorias principales

| Categoria | Paginas | Descripcion |
|---|---|---|
| refformulas | 317 | Lenguaje de formulas |
| admin | 304 | Instalacion, configuracion, VMArgs |
| eecbase | 119 | Funcionalidad base |
| refformui | 84 | Referencia Form-UI |
| refcommands | 79 | Comandos |
| refscripting | 58 | Referencia scripting |
| concept | 58 | Conceptos |
| eececad | 47 | Modulo ECAD |
| eecplc | 47 | Modulo PLC |
| + 27 mas... | ~500 | Tutoriales, SAP, Office, Graph2D, etc. |

## Fuente

Documentacion oficial: https://www.eplan.help/en-us/infoportal/content/eecpro/2026/Content/htm/main_k_home.htm
