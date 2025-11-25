"""
Smart Chunker for Network Documentation and CLI Configuration Text

This module provides syntax-aware and semantic chunking for technical documentation,
particularly optimized for network device configurations (Cisco, Huawei, H3C).

Features:
- Preserves CLI command blocks and configuration contexts
- Respects document structure (headings, sections, paragraphs)
- Token-based chunking with configurable overlap
- Deduplication to prevent identical chunks
- No mid-command or mid-sentence splitting
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""
    min_chunk_size: int = 600      # Minimum tokens per chunk
    max_chunk_size: int = 1200     # Maximum tokens per chunk
    overlap_tokens: int = 100      # Overlap between chunks
    avg_chars_per_token: float = 4.0  # Rough estimate for token counting


class SmartChunker:
    """
    Intelligent text chunker for network documentation and CLI configurations.
    
    Handles:
    - Cisco IOS/IOS-XE/NX-OS configurations
    - Huawei VRP configurations
    - H3C Comware configurations
    - Markdown/structured documentation
    - Mixed content types
    """
    
    # CLI prompt patterns (Cisco, Huawei, H3C)
    CLI_PROMPT_PATTERNS = [
        r'^\[[\w\-]+\]',                    # [SwitchA], [SwitchA-ospf-1]
        r'^[\w\-]+\([\w\-]+\)#',            # Router(config)#, Router(config-if)#
        r'^[\w\-]+#',                        # Router#, Switch#
        r'^[\w\-]+>',                        # Router>, Switch>
        r'^<[\w\-]+>',                       # <Router>, <Switch>
    ]
    
    # Configuration mode markers
    CONFIG_MODE_KEYWORDS = [
        'interface', 'router', 'ospf', 'bgp', 'eigrp', 'rip',
        'vlan', 'access-list', 'route-map', 'policy-map', 'class-map',
        'line', 'aaa', 'snmp-server', 'ntp', 'logging',
        'ip', 'ipv6', 'spanning-tree', 'port-channel', 'lacp',
    ]
    
    # Block terminators
    BLOCK_TERMINATORS = ['exit', 'quit', 'end', '!', '#']
    
    def __init__(self, config: ChunkConfig = None):
        """Initialize the smart chunker with configuration."""
        self.config = config or ChunkConfig()
        self._seen_hashes: Set[str] = set()  # For deduplication
        
        # Compile regex patterns
        self.cli_prompt_re = re.compile('|'.join(self.CLI_PROMPT_PATTERNS), re.MULTILINE)
        self.heading_re = re.compile(r'^(#{1,6}\s+.+|Chapter \d+|Section \d+|CHAPTER .+|SECTION .+)$', re.MULTILINE)
        self.blank_line_re = re.compile(r'\n\s*\n')
        self.list_item_re = re.compile(r'^(\s*[-*•]|\s*\d+[\.)]\s+)', re.MULTILINE)
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return int(len(text) / self.config.avg_chars_per_token)
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for deduplication."""
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        """Check if chunk is duplicate."""
        text_hash = self._hash_text(text)
        if text_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(text_hash)
        return False
    
    def _detect_cli_block_boundaries(self, text: str) -> List[int]:
        """
        Detect CLI configuration block boundaries.
        Returns list of character positions where blocks end.
        """
        boundaries = [0]
        lines = text.split('\n')
        current_pos = 0
        in_config_block = False
        indent_level = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                current_pos += len(line) + 1
                continue
            
            # Detect prompt changes (new command context)
            if self.cli_prompt_re.match(line):
                if current_pos > boundaries[-1]:
                    boundaries.append(current_pos)
                in_config_block = False
                indent_level = 0
            
            # Detect config mode entry
            elif any(line_stripped.startswith(kw) for kw in self.CONFIG_MODE_KEYWORDS):
                if not in_config_block and current_pos > boundaries[-1]:
                    boundaries.append(current_pos)
                in_config_block = True
                indent_level = len(line) - len(line.lstrip())
            
            # Detect block terminators
            elif line_stripped in self.BLOCK_TERMINATORS:
                in_config_block = False
                indent_level = 0
                boundaries.append(current_pos + len(line) + 1)
            
            # Detect dedent (exiting nested config)
            elif in_config_block:
                current_indent = len(line) - len(line.lstrip())
                if current_indent < indent_level and current_pos > boundaries[-1]:
                    boundaries.append(current_pos)
                    indent_level = current_indent
            
            current_pos += len(line) + 1
        
        boundaries.append(len(text))
        return sorted(set(boundaries))
    
    def _detect_document_boundaries(self, text: str) -> List[int]:
        """
        Detect document structure boundaries (headings, paragraphs).
        Returns list of character positions.
        """
        boundaries = [0]
        
        # Heading boundaries
        for match in self.heading_re.finditer(text):
            boundaries.append(match.start())
        
        # Paragraph boundaries (double newlines)
        for match in self.blank_line_re.finditer(text):
            boundaries.append(match.end())
        
        boundaries.append(len(text))
        return sorted(set(boundaries))
    
    def _detect_all_boundaries(self, text: str) -> List[int]:
        """Combine all boundary detection strategies."""
        cli_boundaries = self._detect_cli_block_boundaries(text)
        doc_boundaries = self._detect_document_boundaries(text)
        
        # Merge and deduplicate
        all_boundaries = sorted(set(cli_boundaries + doc_boundaries))
        return all_boundaries
    
    def _find_safe_split_point(self, text: str, start: int, ideal_end: int, max_end: int) -> int:
        """
        Find the safest point to split text between start and max_end,
        preferring positions near ideal_end that respect boundaries.
        """
        # Get boundaries in the relevant range
        boundaries = self._detect_all_boundaries(text[start:max_end])
        boundaries = [start + b for b in boundaries if start < start + b <= max_end]
        
        if not boundaries:
            # Fallback: find sentence or line boundary
            search_text = text[start:max_end]
            
            # Try to split at sentence end
            sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_text)]
            if sentence_ends:
                # Find closest to ideal_end
                best = min(sentence_ends, key=lambda x: abs(start + x - ideal_end))
                return start + best
            
            # Try to split at line break
            line_breaks = [m.end() for m in re.finditer(r'\n', search_text)]
            if line_breaks:
                best = min(line_breaks, key=lambda x: abs(start + x - ideal_end))
                return start + best
            
            # Last resort: split at space
            spaces = [m.end() for m in re.finditer(r'\s+', search_text)]
            if spaces:
                best = min(spaces, key=lambda x: abs(start + x - ideal_end))
                return start + best
            
            return max_end
        
        # Find boundary closest to ideal_end
        best_boundary = min(boundaries, key=lambda x: abs(x - ideal_end))
        
        # Ensure we're making progress
        if best_boundary <= start:
            return max_end
        
        return best_boundary
    
    def _create_chunks_with_overlap(self, text: str) -> List[str]:
        """
        Create chunks with overlap, respecting safe boundaries.
        """
        chunks = []
        text_length = len(text)
        position = 0
        
        min_chars = int(self.config.min_chunk_size * self.config.avg_chars_per_token)
        max_chars = int(self.config.max_chunk_size * self.config.avg_chars_per_token)
        overlap_chars = int(self.config.overlap_tokens * self.config.avg_chars_per_token)
        
        while position < text_length:
            # Calculate chunk boundaries
            ideal_end = position + max_chars
            max_end = min(position + max_chars + 200, text_length)  # Allow slight overflow
            
            # Find safe split point
            if max_end >= text_length:
                split_point = text_length
            else:
                split_point = self._find_safe_split_point(text, position, ideal_end, max_end)
            
            # Extract chunk
            chunk = text[position:split_point].strip()
            
            # Only add if chunk is substantial and not duplicate
            if chunk and len(chunk) > 50 and not self._is_duplicate(chunk):
                chunks.append(chunk)
            
            # Calculate next position with overlap
            if split_point >= text_length:
                break
            
            # Move forward, but overlap by going back
            next_position = split_point - overlap_chars
            
            # Ensure we make progress
            if next_position <= position:
                next_position = split_point
            
            # Try to find a good overlap point (start of sentence/line)
            if next_position < split_point:
                overlap_text = text[next_position:split_point]
                # Find first sentence or line start in overlap region
                sentence_starts = [m.start() for m in re.finditer(r'(?:^|\n|[.!?]\s+)([A-Z\[<])', overlap_text)]
                if sentence_starts and sentence_starts[0] > 0:
                    next_position += sentence_starts[0]
            
            position = next_position
        
        return chunks
    
    def chunk(self, text: str, source: str = "unknown") -> List[Dict[str, any]]:
        """
        Chunk text intelligently with metadata.
        
        Args:
            text: The text to chunk
            source: Source identifier (filename, document name)
        
        Returns:
            List of dictionaries with 'source', 'chunk_index', and 'text'
        """
        if not text or not text.strip():
            return []
        
        # Reset deduplication for new document
        self._seen_hashes.clear()
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Create chunks
        chunk_texts = self._create_chunks_with_overlap(cleaned_text)
        
        # Format output
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append({
                "source": source,
                "chunk_index": idx,
                "text": chunk_text
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text while preserving important formatting.
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Preserve indentation but normalize spaces
        lines = []
        for line in text.split('\n'):
            # Keep leading whitespace, strip trailing
            if line.strip():
                lines.append(line.rstrip())
            else:
                lines.append('')
        
        return '\n'.join(lines)


def chunk_text(text: str, source: str = "unknown", config: ChunkConfig = None) -> List[Dict[str, any]]:
    """
    Convenience function for chunking text.
    
    Args:
        text: Text to chunk
        source: Source identifier
        config: Optional ChunkConfig instance
    
    Returns:
        List of chunk dictionaries
    """
    chunker = SmartChunker(config)
    return chunker.chunk(text, source)


# Backward compatibility with old interface
def chunk_text_simple(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Simple interface matching old chunker API.
    Returns list of chunk texts (no metadata).
    """
    config = ChunkConfig(
        min_chunk_size=int(chunk_size * 0.7 / 4),
        max_chunk_size=int(chunk_size / 4),
        overlap_tokens=int(overlap / 4)
    )
    chunker = SmartChunker(config)
    chunks = chunker.chunk(text, "")
    return [c['text'] for c in chunks]


if __name__ == "__main__":
    # Example usage
    sample_config = """
    interface GigabitEthernet0/0/1
     description Uplink to Core
     ip address 192.168.1.1 255.255.255.0
     ospf network-type broadcast
     ospf priority 100
    quit
    
    router ospf 1
     area 0.0.0.0
     network 192.168.1.0 0.0.0.255
     network 10.0.0.0 0.255.255.255
    quit
    
    [SwitchA]vlan 10
    [SwitchA-vlan10]description Management VLAN
    [SwitchA-vlan10]quit
    """
    
    chunker = SmartChunker()
    chunks = chunker.chunk(sample_config, "test.txt")
    
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n--- Chunk {chunk['chunk_index']} ---")
        print(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
