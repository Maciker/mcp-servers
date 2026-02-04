"""
Database Initialization Script for Papers MCP Server

This script automatically indexes all PDF files present in the ./papers directory.
Run this script once to populate the ChromaDB vector database before using the MCP server.

Usage:
    uv run python init_db.py

Author: Generated for University PoC
License: MIT
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server import (
    PAPERS_DIR,
    CHROMA_DB_DIR,
    collection,
    chunk_text,
    extract_doi,
    generate_embeddings,
    pymupdf4llm
)


def get_indexed_papers() -> set[str]:
    """
    Get the set of filenames already indexed in ChromaDB.
    
    Returns:
        Set of indexed filenames
    """
    try:
        # Get all unique filenames from the collection
        results = collection.get(include=["metadatas"])
        if results and results['metadatas']:
            return {m['filename'] for m in results['metadatas'] if 'filename' in m}
    except Exception:
        pass
    return set()


def index_paper(pdf_path: Path, skip_existing: bool = True) -> dict:
    """
    Index a single PDF file into ChromaDB.
    
    Args:
        pdf_path: Path to the PDF file
        skip_existing: Whether to skip already-indexed papers
        
    Returns:
        Dictionary with indexing results
    """
    filename = pdf_path.name
    result = {
        "filename": filename,
        "status": "unknown",
        "chunks": 0,
        "doi": None,
        "error": None
    }
    
    try:
        # Check if already indexed
        if skip_existing:
            existing = collection.get(
                where={"filename": filename},
                limit=1
            )
            if existing and existing['ids']:
                result["status"] = "skipped"
                result["chunks"] = len(existing['ids'])
                return result
        
        # Extract text from PDF
        print(f"  📄 Extracting text from {filename}...")
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        if not md_text or len(md_text.strip()) < 100:
            result["status"] = "error"
            result["error"] = "Could not extract meaningful text"
            return result
        
        # Extract DOI
        doi = extract_doi(md_text)
        result["doi"] = doi
        
        # Split into chunks
        print(f"  📝 Splitting into chunks...")
        chunks = chunk_text(md_text)
        
        if not chunks:
            result["status"] = "error"
            result["error"] = "Could not create text chunks"
            return result
        
        result["chunks"] = len(chunks)
        
        # Generate embeddings
        print(f"  🧠 Generating {len(chunks)} embeddings locally...")
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
        print(f"  💾 Storing in ChromaDB...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    """
    Main entry point for the initialization script.
    Indexes all PDFs in the papers directory.
    """
    print("=" * 60)
    print("📚 Papers MCP - Database Initialization")
    print("=" * 60)
    print()
    
    # Check papers directory
    if not PAPERS_DIR.exists():
        print(f"❌ Papers directory not found: {PAPERS_DIR}")
        print(f"   Please create the directory and add PDF files.")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {PAPERS_DIR}")
        sys.exit(1)
    
    print(f"📁 Papers directory: {PAPERS_DIR}")
    print(f"📁 ChromaDB location: {CHROMA_DB_DIR}")
    print(f"📄 Found {len(pdf_files)} PDF files")
    print()
    
    # Get already indexed papers
    indexed = get_indexed_papers()
    print(f"📊 Already indexed: {len(indexed)} papers")
    print()
    
    # Process each PDF
    results = {
        "success": [],
        "skipped": [],
        "error": []
    }
    
    start_time = datetime.now()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        result = index_paper(pdf_path)
        results[result["status"]].append(result)
        
        if result["status"] == "success":
            print(f"  ✅ Success: {result['chunks']} chunks, DOI: {result['doi'] or 'Not found'}")
        elif result["status"] == "skipped":
            print(f"  ⏭️  Skipped: Already indexed")
        else:
            print(f"  ❌ Error: {result['error']}")
        print()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"✅ Successfully indexed: {len(results['success'])} papers")
    print(f"⏭️  Skipped (existing):  {len(results['skipped'])} papers")
    print(f"❌ Errors:              {len(results['error'])} papers")
    print(f"⏱️  Total time:          {elapsed:.1f} seconds")
    print()
    
    # Final collection stats
    total_docs = collection.count()
    print(f"📚 Total documents in database: {total_docs}")
    print()
    
    if results['error']:
        print("⚠️  Papers with errors:")
        for r in results['error']:
            print(f"   - {r['filename']}: {r['error']}")
        print()
    
    print("🚀 Database initialization complete!")
    print("   Run 'uv run fastmcp dev server.py' to start the MCP server.")


if __name__ == "__main__":
    main()
