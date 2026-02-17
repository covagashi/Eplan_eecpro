# EPLAN EEC Pro 2026 - Documentation RAG

Scraper + RAG (Retrieval Augmented Generation) de la documentacion oficial de **EPLAN Engineering Configuration Pro 2026**.

## Contenido

- `docs_md/` - 1648 paginas de documentacion convertidas a Markdown, organizadas en 36 categorias
- `rag_db/` - Base de datos vectorial ChromaDB con 1967 chunks indexados
- `scrape_eplan.py` - Script de scraping de la documentacion
- `rag_index.py` - Indexador de documentos en ChromaDB
- `rag_query.py` - Buscador semantico por query

## Uso rapido

```bash
# Instalar dependencias
pip install requests beautifulsoup4 html2text chromadb

# Re-indexar (solo si se modifican los docs)
python rag_index.py

# Buscar en la documentacion
python rag_query.py "how to configure Job Server REST API" --top 5
python rag_query.py "GenerateEcadCodeCommand" --top 3
python rag_query.py "formula if then else" --category refformulas
```

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
