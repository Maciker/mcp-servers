# Papers MCP - Local RAG for Research Papers

A fully local and private RAG (Retrieval-Augmented Generation) system using the Model Context Protocol (MCP) to interact with a library of research papers (PDFs).

## Features

- **🔒 Privacy-First**: All embeddings generated locally using `sentence-transformers` (no paper content sent to external APIs)
- **📚 Semantic Search**: Find relevant passages across your entire paper library
- **📝 APA Citations**: Automatically generate APA 7th edition citations
- **🗃️ Persistent Storage**: ChromaDB stores your indexed papers locally

## Quick Start

### 1. Install Dependencies with UV

```powershell
# Install uv if not already installed
# See: https://docs.astral.sh/uv/getting-started/installation/

# Sync dependencies
cd c:\Users\imacaya\Learning\papers-mcp
uv sync
```

### 2. Add Your PDFs

Place your research papers in the `./papers` folder:

```
papers-mcp/
├── papers/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
```

### 3. Initialize the Database

Index all PDFs in the papers folder:

```powershell
uv run python init_db.py
```

This will:
- Extract text from each PDF (preserving tables and structure)
- Split into semantic chunks (~500 tokens each)
- Generate embeddings locally
- Store in ChromaDB at `./chroma_db`

### 4. Configure Claude Desktop

Add to your `claude_desktop_config.json`:

**Windows** (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "papers-library": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "c:\\Users\\imacaya\\Learning\\papers-mcp",
        "python",
        "server.py"
      ]
    }
  }
}
```

**macOS** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "papers-library": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/papers-mcp",
        "python",
        "server.py"
      ]
    }
  }
}
```

### 5. Restart Claude Desktop

Restart the application to load the new MCP server.

## Available Tools

### `ingest_paper(filename)`

Index a single PDF file into the database.

```
Example: ingest_paper("research_paper.pdf")
```

### `search_library(query, n_results=5)`

Perform semantic search across all indexed papers.

```
Example: search_library("machine learning for image classification")
```

Returns the 3-5 most relevant text chunks with source information.

### `generate_apa_citation(filename)`

Generate an APA 7th edition citation.

```
Example: generate_apa_citation("research_paper.pdf")
```

Uses DOI to fetch metadata from CrossRef if available.

## Development

### Test the Server Locally

```powershell
# Interactive development mode
uv run fastmcp dev server.py

# Run directly
uv run python server.py
```

### Dependency Management with UV

```powershell
# Add a new dependency
uv add package-name

# Update dependencies
uv sync --upgrade

# Run any script
uv run python script.py
```

## Architecture

```
papers-mcp/
├── server.py        # MCP server with FastMCP (3 tools)
├── init_db.py       # Batch indexing script
├── papers/          # Your PDF files
├── chroma_db/       # Persistent vector database
├── pyproject.toml   # Dependencies
└── README.md        # This file
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| MCP Framework | FastMCP |
| PDF Extraction | pymupdf4llm |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Metadata Lookup | CrossRef API |

## Privacy

This system is designed for **total privacy**:

- ✅ Text extraction: Local (pymupdf4llm)
- ✅ Embedding generation: Local (sentence-transformers)
- ✅ Vector storage: Local (ChromaDB)
- ✅ Search: Local (ChromaDB similarity)
- ⚠️ Citation metadata: CrossRef API (DOI only, no paper content)

No paper content is ever sent to external APIs.

## License

MIT
