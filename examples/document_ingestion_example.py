"""
Document Ingestion Example

This example demonstrates the document ingestion feature in Prompture,
showing how to parse various document formats (PDF, DOCX, HTML, Markdown, CSV, Excel)
into text suitable for LLM extraction.

Key features demonstrated:
- Basic ingest() usage with auto file type detection
- CSV parsing with structured data
- File type override
- Document chunking for large documents
- Integration with extract_with_model() via source= parameter
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel

from prompture import extract_with_model, field_from_registry
from prompture.ingestion import ChunkingConfig, chunk_document, ingest

# =============================================================================
# Example 1: Basic Markdown Ingestion
# =============================================================================


def example_basic_markdown():
    """Demonstrate basic markdown file ingestion."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Markdown Ingestion")
    print("=" * 60)

    # Create a sample markdown file
    markdown_content = """# Product Report

## Executive Summary
This quarter showed strong growth in the enterprise segment.

## Key Metrics
- Revenue: $2.5M
- Customer acquisition: 150 new clients
- Retention rate: 94%

## Highlights
The **AI-powered analytics** feature drove significant engagement.
"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(markdown_content)
        temp_path = f.name

    try:
        # Ingest the markdown file
        doc = ingest(temp_path)

        print(f"File type: {doc.file_type}")
        print(f"Source: {doc.source_path}")
        print(f"Character count: {doc.char_count}")
        print(f"\nExtracted text (first 200 chars):")
        print(doc.text[:200])
        print("\n✅ Successfully ingested markdown file")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Example 2: CSV Ingestion with Structured Data
# =============================================================================


def example_csv_ingestion():
    """Demonstrate CSV file ingestion with metadata extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CSV Ingestion")
    print("=" * 60)

    # Create a sample CSV file
    csv_content = """product_name,price,category,stock
Laptop Pro,1299.99,Electronics,45
Office Chair,349.50,Furniture,120
Coffee Maker,89.99,Appliances,78
Desk Lamp,34.99,Furniture,200
"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(csv_content)
        temp_path = f.name

    try:
        # Ingest the CSV file
        doc = ingest(temp_path)

        print(f"File type: {doc.file_type}")
        print(f"Row count: {doc.metadata.get('row_count')}")
        print(f"Headers: {doc.metadata.get('headers')}")
        print(f"Delimiter: {repr(doc.metadata.get('delimiter'))}")
        print(f"\nExtracted data (first 150 chars):")
        print(doc.text[:150])
        print("\n✅ Successfully ingested CSV file with metadata")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Example 3: File Type Override
# =============================================================================


def example_file_type_override():
    """Demonstrate explicit file type override."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: File Type Override")
    print("=" * 60)

    # Create a text file that we'll parse as markdown
    content = """# Notes

This is a plain .txt file, but we want to parse it as Markdown.

## Benefits
- Explicit control over parser selection
- Handle files with ambiguous extensions
- Process data streams with known format
"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(content)
        temp_path = f.name

    try:
        # Without override, .txt defaults to markdown parser
        doc_default = ingest(temp_path)
        print(f"Default detection: {doc_default.file_type}")

        # Explicit override (same result in this case, but demonstrates the API)
        doc_override = ingest(temp_path, file_type="markdown")
        print(f"With override: {doc_override.file_type}")

        print(f"\nExtracted text (first 100 chars):")
        print(doc_override.text[:100])
        print("\n✅ File type override works as expected")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Example 4: Document Chunking for Large Documents
# =============================================================================


def example_document_chunking():
    """Demonstrate chunking large documents."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Document Chunking")
    print("=" * 60)

    # Create a long document
    paragraphs = [
        f"This is paragraph {i}. " * 20  # Each para ~400 chars
        for i in range(10)
    ]
    long_text = "\n\n".join(paragraphs)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(long_text)
        temp_path = f.name

    try:
        # Ingest the document
        doc = ingest(temp_path)
        print(f"Total document size: {doc.char_count} characters")

        # Chunk the document
        config = ChunkingConfig(
            max_chars_per_chunk=1000,
            overlap_chars=100,
            strategy="chars",
        )
        chunks = chunk_document(doc, config)

        print(f"Number of chunks: {len(chunks)}")
        print("\nChunk details:")
        for chunk in chunks:
            print(f"  Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}: "
                  f"{len(chunk.text)} chars")
            print(f"    First 50 chars: {chunk.text[:50]}...")

        print("\n✅ Successfully chunked large document")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Example 5: Integration with extract_with_model (Commented)
# =============================================================================


def example_extract_with_model_integration():
    """Demonstrate source= parameter with extract_with_model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Integration with extract_with_model()")
    print("=" * 60)

    print("The source= parameter allows direct document ingestion")
    print("during extraction. Here's how it would work:\n")

    print("Example code (requires LLM API key):")
    print("-" * 40)
    print("""
# Define extraction model
class ProductInfo(BaseModel):
    name: str = field_from_registry("name")
    price: str = field_from_registry("price")
    category: str = field_from_registry("category")

# Extract directly from document
result = extract_with_model(
    ProductInfo,
    model_name="openai/gpt-4",
    source="product_description.pdf",  # Path to PDF file
)

print(result.model.name)
print(result.model.price)
""")
    print("-" * 40)

    # Create a sample document to show the API
    sample_text = """Product Name: SuperWidget Pro
Price: $299.99
Category: Electronics

The SuperWidget Pro is our flagship product featuring
advanced AI capabilities and premium build quality.
"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(sample_text)
        temp_path = f.name

    try:
        # Show the ingestion part (extraction would need LLM)
        doc = ingest(temp_path)
        print(f"\nSample document ingested:")
        print(f"  Type: {doc.file_type}")
        print(f"  Size: {doc.char_count} chars")
        print(f"  Content preview:")
        print(f"  {doc.text[:100]}...")

        print("\n💡 To actually extract, you would use:")
        print("   result = extract_with_model(ProductInfo, ")
        print(f"                                model_name='openai/gpt-4',")
        print(f"                                source='{temp_path}')")
        print("\n✅ source= parameter ready for use with LLM extraction")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Example 6: Working with Different File Types
# =============================================================================


def example_multiple_file_types():
    """Show ingestion across different file types."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Multiple File Type Support")
    print("=" * 60)

    file_types = {
        ".md": "# Markdown\n\nMarkdown content here.",
        ".csv": "col1,col2\nval1,val2",
        ".txt": "Plain text content.",
    }

    print("Supported file types and their parsers:\n")

    for ext, content in file_types.items():
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=ext) as f:
            f.write(content)
            temp_path = f.name

        try:
            doc = ingest(temp_path)
            print(f"  {ext:6} → {doc.file_type:10} parser "
                  f"({doc.char_count} chars extracted)")
        finally:
            Path(temp_path).unlink()

    print("\n📋 Also supported (with optional dependencies):")
    print("  .pdf   → pdf        parser (requires PyPDF2 or pypdf)")
    print("  .docx  → docx       parser (requires python-docx)")
    print("  .html  → html       parser (requires beautifulsoup4)")
    print("  .xlsx  → xlsx       parser (requires openpyxl)")

    print("\n✅ Multiple file formats supported")


# =============================================================================
# Example 7: Advanced Chunking Strategies
# =============================================================================


def example_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Chunking Strategies")
    print("=" * 60)

    # Create document with page-like structure
    content_with_pages = f"Page 1: {'text ' * 100}\n\nPage 2: {'text ' * 100}"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(content_with_pages)
        temp_path = f.name

    try:
        doc = ingest(temp_path)

        # Strategy 1: Character-based chunking
        print("Strategy 1: Character-based chunking")
        config_chars = ChunkingConfig(
            max_chars_per_chunk=500,
            overlap_chars=50,
            strategy="chars",
        )
        chunks_chars = chunk_document(doc, config_chars)
        print(f"  Created {len(chunks_chars)} chunks with char-based strategy")
        print(f"  Each chunk ~{config_chars.max_chars_per_chunk} chars max")
        print(f"  Overlap: {config_chars.overlap_chars} chars\n")

        # Strategy 2: Page-based chunking (fallback to chars if no pages)
        print("Strategy 2: Page-based chunking")
        config_pages = ChunkingConfig(
            max_chars_per_chunk=800,
            strategy="page",
        )
        chunks_pages = chunk_document(doc, config_pages)
        print(f"  Created {len(chunks_pages)} chunks with page-based strategy")
        print(f"  (Falls back to char-based for non-paginated docs)")

        print("\n✅ Both chunking strategies demonstrated")
    finally:
        Path(temp_path).unlink()


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Run all document ingestion examples."""
    print("🚀 Prompture Document Ingestion Examples\n")

    # Run all examples
    example_basic_markdown()
    example_csv_ingestion()
    example_file_type_override()
    example_document_chunking()
    example_extract_with_model_integration()
    example_multiple_file_types()
    example_chunking_strategies()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Document ingestion feature demonstrated successfully!\n")

    print("📚 Key Takeaways:")
    print("  1. ingest() auto-detects file types from extensions")
    print("  2. CSV/TSV parsing extracts metadata (headers, row count)")
    print("  3. file_type= parameter allows explicit parser selection")
    print("  4. chunk_document() handles large documents efficiently")
    print("  5. source= parameter integrates with extract_with_model()")
    print("  6. Multiple file formats supported with optional deps")
    print("  7. Two chunking strategies: char-based and page-based\n")

    print("🔧 Installation:")
    print("  Basic (markdown, csv): pip install prompture")
    print("  Full support: pip install prompture[ingest]\n")

    print("📖 Next Steps:")
    print("  - Try ingesting your own documents")
    print("  - Combine with extract_with_model() for LLM extraction")
    print("  - Experiment with chunking for large files")
    print("  - Register custom parsers for specialized formats")


if __name__ == "__main__":
    main()
