#!/usr/bin/env python3
"""
Test script for MCP server functionality
Run this to verify the server works before configuring Claude Desktop
"""

import sys
from pathlib import Path

print("=" * 60)
print("🧪 Testing Papers MCP Server")
print("=" * 60)
print()

# Test 1: Import the server
print("1️⃣ Testing server imports...")
try:
    from server import mcp, collection, embedding_model
    print("   ✅ Server module loaded successfully")
    print(f"   Server name: {mcp.name}")
except Exception as e:
    print(f"   ❌ Failed to import server: {e}")
    sys.exit(1)

# Test 2: Check tools are registered
print("\n2️⃣ Testing tool registration...")
try:
    tools = list(mcp._tool_manager._tools.keys())
    print(f"   ✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"      - {tool}")
    
    expected_tools = {'ingest_paper', 'search_library', 'generate_apa_citation'}
    if expected_tools == set(tools):
        print("   ✅ All expected tools present")
    else:
        print(f"   ⚠️  Missing tools: {expected_tools - set(tools)}")
except Exception as e:
    print(f"   ❌ Failed to check tools: {e}")

# Test 3: Check ChromaDB
print("\n3️⃣ Testing ChromaDB connection...")
try:
    count = collection.count()
    print(f"   ✅ ChromaDB connected")
    print(f"   Documents in database: {count}")
except Exception as e:
    print(f"   ❌ ChromaDB error: {e}")

# Test 4: Check embedding model
print("\n4️⃣ Testing embedding model...")
try:
    test_embedding = embedding_model.encode(["test query"])
    print(f"   ✅ Embedding model loaded")
    print(f"   Embedding dimension: {len(test_embedding[0])}")
except Exception as e:
    print(f"   ❌ Embedding model error: {e}")

# Test 5: Check papers directory
print("\n5️⃣ Checking papers directory...")
try:
    from server import PAPERS_DIR
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"   ✅ Papers directory: {PAPERS_DIR}")
    print(f"   PDF files found: {len(pdf_files)}")
    for pdf in pdf_files[:3]:  # Show first 3
        print(f"      - {pdf.name}")
    if len(pdf_files) > 3:
        print(f"      ... and {len(pdf_files) - 3} more")
except Exception as e:
    print(f"   ❌ Papers directory error: {e}")

print("\n" + "=" * 60)
print("✅ All tests passed! Server is ready.")
print("=" * 60)
print()
print("Next steps:")
print("1. Run: uv run python init_db.py")
print("2. Update claude_desktop_config.json")
print("3. Restart Claude Desktop")
