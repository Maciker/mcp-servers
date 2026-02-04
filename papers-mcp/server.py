"""
MCP Server for Research Papers RAG

A fully local and private RAG system using the Model Context Protocol (MCP)
to interact with a library of research papers (PDFs).

Author: Generated for University PoC
License: MIT
"""

import os
import re
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
import chromadb
from sentence_transformers import SentenceTransformer
import pymupdf4llm
from crossref.restful import Works

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "papers"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Chunking parameters
CHUNK_SIZE = 500  # Target tokens per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
CHARS_PER_TOKEN = 4  # Approximate characters per token

# DOI regex pattern
DOI_PATTERN = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)

# =============================================================================
# Initialize Components
# =============================================================================

# ChromaDB with local persistence
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
)

# Local embedding model (runs on CPU, no external API calls)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# CrossRef API client for metadata lookup
works = Works()

# FastMCP Server
mcp = FastMCP(
    name="Papers Library",
    instructions="""
    This MCP server provides tools to interact with a local library of research papers.
    
    Available tools:
    - ingest_paper: Index a PDF file into the vector database
    - search_library: Perform semantic search across all indexed papers
    - generate_apa_citation: Generate APA 7th edition citation for a paper
    
    All processing is done locally - no paper content is sent to external APIs.
    """
)

# =============================================================================
# Helper Functions
# =============================================================================


def extract_doi(text: str) -> Optional[str]:
    """
    Extract DOI from text using regex pattern.
    
    Args:
        text: Text content to search for DOI
        
    Returns:
        DOI string if found, None otherwise
    """
    match = DOI_PATTERN.search(text)
    if match:
        doi = match.group(0)
        # Clean up trailing punctuation that might be captured
        doi = doi.rstrip('.,;:)')
        return doi
    return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into semantically meaningful chunks with overlap.
    
    Uses paragraph boundaries when possible to maintain coherence.
    
    Args:
        text: Text to split into chunks
        chunk_size: Target number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    target_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_length = len(para)
        
        # If paragraph alone exceeds chunk size, split it further
        if para_length > target_chars:
            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > target_chars and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Keep overlap from the end of current chunk
                    overlap_text = ' '.join(current_chunk)[-overlap_chars:]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text) if overlap_text else 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
        else:
            # Add paragraph to current chunk
            if current_length + para_length > target_chars and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap from the end of current chunk
                overlap_text = ' '.join(current_chunk)[-overlap_chars:]
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(para)
            current_length += para_length
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the local model.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def fetch_crossref_metadata(doi: str) -> Optional[dict]:
    """
    Fetch paper metadata from CrossRef API using DOI.
    
    Args:
        doi: DOI string
        
    Returns:
        Dictionary with metadata or None if not found
    """
    try:
        work = works.doi(doi)
        if work:
            return {
                'title': work.get('title', ['Unknown'])[0] if work.get('title') else 'Unknown',
                'authors': work.get('author', []),
                'year': work.get('published-print', {}).get('date-parts', [[None]])[0][0] 
                        or work.get('published-online', {}).get('date-parts', [[None]])[0][0],
                'journal': work.get('container-title', [''])[0] if work.get('container-title') else '',
                'volume': work.get('volume', ''),
                'issue': work.get('issue', ''),
                'pages': work.get('page', ''),
                'doi': doi
            }
    except Exception:
        pass
    return None


def format_apa_citation(metadata: dict) -> str:
    """
    Format metadata into APA 7th edition citation.
    
    Args:
        metadata: Dictionary with paper metadata
        
    Returns:
        Formatted APA citation string
    """
    # Format authors
    authors = metadata.get('authors', [])
    if authors:
        author_strs = []
        for author in authors[:7]:  # APA limits to 7 authors
            family = author.get('family', '')
            given = author.get('given', '')
            if given:
                initials = '. '.join([name[0] for name in given.split()]) + '.'
                author_strs.append(f"{family}, {initials}")
            else:
                author_strs.append(family)
        
        if len(authors) > 7:
            author_str = ', '.join(author_strs[:6]) + ', ... ' + author_strs[-1]
        elif len(authors) == 2:
            author_str = ' & '.join(author_strs)
        elif len(authors) > 2:
            author_str = ', '.join(author_strs[:-1]) + ', & ' + author_strs[-1]
        else:
            author_str = author_strs[0] if author_strs else 'Unknown Author'
    else:
        author_str = 'Unknown Author'
    
    # Build citation
    year = metadata.get('year', 'n.d.')
    title = metadata.get('title', 'Unknown Title')
    journal = metadata.get('journal', '')
    volume = metadata.get('volume', '')
    issue = metadata.get('issue', '')
    pages = metadata.get('pages', '')
    doi = metadata.get('doi', '')
    
    citation = f"{author_str} ({year}). {title}."
    
    if journal:
        citation += f" *{journal}*"
        if volume:
            citation += f", *{volume}*"
        if issue:
            citation += f"({issue})"
        if pages:
            citation += f", {pages}"
        citation += "."
    
    if doi:
        citation += f" https://doi.org/{doi}"
    
    return citation


# =============================================================================
# MCP Tools
# =============================================================================


@mcp.tool()
def ingest_paper(filename: str) -> str:
    """
    Read a PDF from the papers folder, process it, and store it in the vector database.
    
    This tool extracts text from the PDF while preserving tables and academic structures,
    splits it into semantic chunks, generates embeddings locally, and stores everything
    in ChromaDB with metadata including the filename and DOI (if found).
    
    Args:
        filename: Name of the PDF file (e.g., "research_paper.pdf")
        
    Returns:
        Success message with processing details or error message
    """
    try:
        # Validate file path
        pdf_path = PAPERS_DIR / filename
        
        if not pdf_path.exists():
            return f"Error: File '{filename}' not found in papers directory."
        
        if not filename.lower().endswith('.pdf'):
            return f"Error: File '{filename}' is not a PDF file."
        
        # Check if already indexed
        existing = collection.get(
            where={"filename": filename},
            limit=1
        )
        if existing and existing['ids']:
            return f"Paper '{filename}' is already indexed. Use search_library to query it."
        
        # Extract text from PDF using pymupdf4llm (preserves tables and structure)
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        if not md_text or len(md_text.strip()) < 100:
            return f"Error: Could not extract meaningful text from '{filename}'."
        
        # Extract DOI from the text
        doi = extract_doi(md_text)
        
        # Split into chunks
        chunks = chunk_text(md_text)
        
        if not chunks:
            return f"Error: Could not create text chunks from '{filename}'."
        
        # Generate embeddings locally
        embeddings = generate_embeddings(chunks)
        
        # Prepare data for ChromaDB
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "filename": filename,
                "doi": doi or "",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        return (
            f"Successfully ingested '{filename}':\n"
            f"- Extracted {len(md_text):,} characters\n"
            f"- Created {len(chunks)} chunks\n"
            f"- DOI: {doi or 'Not found'}\n"
            f"- All embeddings generated locally (privacy preserved)"
        )
        
    except Exception as e:
        return f"Error ingesting '{filename}': {str(e)}"


@mcp.tool()
def search_library(query: str, n_results: int = 5) -> list[dict]:
    """
    Perform semantic search across all indexed papers.
    
    This tool generates an embedding for your query using the local model,
    then finds the most similar chunks in the vector database.
    
    Args:
        query: Natural language search query (e.g., "machine learning for image classification")
        n_results: Number of results to return (default: 5, max: 10)
        
    Returns:
        List of relevant text chunks with source information
    """
    try:
        # Validate parameters
        if not query or len(query.strip()) < 3:
            return [{"error": "Query must be at least 3 characters long."}]
        
        n_results = min(max(n_results, 1), 10)  # Clamp between 1 and 10
        
        # Check if we have any documents
        count = collection.count()
        if count == 0:
            return [{"error": "No papers indexed yet. Use ingest_paper to add papers first."}]
        
        # Generate query embedding locally
        query_embedding = generate_embeddings([query])[0]
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "rank": i + 1,
                "filename": results['metadatas'][0][i]['filename'],
                "chunk_index": results['metadatas'][0][i]['chunk_index'],
                "total_chunks": results['metadatas'][0][i]['total_chunks'],
                "doi": results['metadatas'][0][i]['doi'] or None,
                "relevance_score": round(1 - results['distances'][0][i], 4),  # Convert distance to similarity
                "content": results['documents'][0][i][:1000] + "..." if len(results['documents'][0][i]) > 1000 else results['documents'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@mcp.tool()
def generate_apa_citation(filename: str) -> str:
    """
    Generate an APA 7th edition citation for an indexed paper.
    
    This tool looks up the paper in the database, retrieves its DOI (if available),
    fetches metadata from CrossRef, and formats a proper APA citation.
    
    Args:
        filename: Name of the PDF file (e.g., "research_paper.pdf")
        
    Returns:
        APA 7th edition formatted citation or error message
    """
    try:
        # Check if paper exists in database
        existing = collection.get(
            where={"filename": filename},
            limit=1,
            include=["metadatas"]
        )
        
        if not existing or not existing['ids']:
            return f"Error: Paper '{filename}' is not indexed. Use ingest_paper first."
        
        # Get DOI from metadata
        doi = existing['metadatas'][0].get('doi', '')
        
        if not doi:
            # Fallback citation without DOI
            return (
                f"Could not generate full APA citation - no DOI found in '{filename}'.\n\n"
                f"Suggested format:\n"
                f"Author(s). (Year). *{filename.replace('.pdf', '')}*. Publisher."
            )
        
        # Fetch metadata from CrossRef (only external API call - for metadata only)
        metadata = fetch_crossref_metadata(doi)
        
        if not metadata:
            return (
                f"Found DOI ({doi}) but could not retrieve metadata from CrossRef.\n\n"
                f"Manual citation with DOI:\n"
                f"Author(s). (Year). Title. *Journal*. https://doi.org/{doi}"
            )
        
        # Format APA citation
        citation = format_apa_citation(metadata)
        
        return (
            f"**APA 7th Edition Citation:**\n\n"
            f"{citation}\n\n"
            f"---\n"
            f"*Source: CrossRef metadata for DOI {doi}*"
        )
        
    except Exception as e:
        return f"Error generating citation for '{filename}': {str(e)}"


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()
