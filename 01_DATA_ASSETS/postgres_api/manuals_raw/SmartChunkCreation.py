"""
Smart Chunk Creation Pipeline

Updated version of ChunkCreationV4.py that uses smart_chunker
for syntax-aware and semantic chunking of network documentation.
"""

import os
import json
from smart_chunker import SmartChunker, ChunkConfig


def clean_text(text: str) -> str:
    """Basic text cleaning (kept for compatibility)."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def process_directory_with_smart_chunker(
    input_dir: str,
    output_file: str,
    config: ChunkConfig = None,
    log_every: int = 50
):
    """
    Process directory of text files using SmartChunker.
    
    Args:
        input_dir: Directory containing .txt files
        output_file: Output JSON file path
        config: ChunkConfig instance (optional)
        log_every: Log progress every N chunks
    """
    if config is None:
        config = ChunkConfig(
            min_chunk_size=600,
            max_chunk_size=1200,
            overlap_tokens=100
        )
    
    chunker = SmartChunker(config)
    total_chunks = 0
    total_files = 0
    first = True
    
    print(f"🔍 Scanning directory: {input_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("[\n")
        
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                print(f"\n📄 Processing {filename}...")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                    continue
                
                # Basic cleaning (remove excessive whitespace)
                cleaned = clean_text(text)
                
                if not cleaned:
                    print(f"⚠️  Skipping empty file: {filename}")
                    continue
                
                # Use smart chunker
                chunks = chunker.chunk(cleaned, source=filename)
                
                print(f"   Generated {len(chunks)} chunks from {filename}")
                
                # Write chunks to JSON
                for chunk_data in chunks:
                    if not first:
                        out.write(",\n")
                    json.dump(chunk_data, out, ensure_ascii=False, indent=2)
                    first = False
                    
                    total_chunks += 1
                    if total_chunks % log_every == 0:
                        print(f"   💾 {total_chunks} chunks saved so far...")
                
                total_files += 1
        
        out.write("\n]\n")
    
    print(f"\n✅ Done! Processed {total_files} files")
    print(f"✅ Created {total_chunks} smart chunks")
    print(f"✅ Output saved to: {output_file}")


def process_directory_json(
    input_dir: str,
    output_file: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    log_every: int = 50
):
    """
    Wrapper function for backward compatibility with old interface.
    Converts old parameters to new ChunkConfig.
    """
    # Convert character-based sizes to token-based config
    config = ChunkConfig(
        min_chunk_size=int(chunk_size * 0.7 / 4),  # chars -> tokens
        max_chunk_size=int(chunk_size / 4),
        overlap_tokens=int(overlap / 4)
    )
    
    process_directory_with_smart_chunker(
        input_dir,
        output_file,
        config,
        log_every
    )


def main():
    """Main entry point."""
    print("=" * 60)
    print("🧠 Smart Chunk Creation Pipeline")
    print("=" * 60)
    
    # Configuration
    input_dir = "./manuals_raw/docs/"
    output_file = "./manuals_raw/docs/raw_chunks.json"
    
    # Smart chunker configuration
    config = ChunkConfig(
        min_chunk_size=600,      # ~600 tokens (2400 chars)
        max_chunk_size=1200,     # ~1200 tokens (4800 chars)
        overlap_tokens=100       # ~100 tokens (400 chars)
    )
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    # Process files
    process_directory_with_smart_chunker(
        input_dir=input_dir,
        output_file=output_file,
        config=config,
        log_every=50
    )
    
    # Verify output
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"📊 Output file size: {file_size:.2f} KB")


if __name__ == "__main__":
    main()
