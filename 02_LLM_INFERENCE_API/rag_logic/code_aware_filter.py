import re
from typing import List, Dict


def filter_and_refine_context(procedural_chunks: list[str], aggressive: bool = False) -> str:
    """
    Filters and refines procedural context chunks using code-aware heuristics.
    
    Args:
        procedural_chunks: List of raw documentation/example chunks from FAISS
        aggressive: If True, applies stricter filtering (higher CLI density requirement)
    
    Returns:
        Single string containing filtered, high-quality CLI context ready for RAG
    
    Filtering Strategy:
        - Heuristic 1: Keep chunks with ≥2 executable CLI command lines (≥3 if aggressive)
        - Heuristic 2: Remove non-CLI explanatory text and comments
        - Heuristic 3: Preserve indentation and command context
        - Stitch remaining chunks with clear separators
    """
    
    min_cli_commands = 3 if aggressive else 2
    
    # Cisco/Huawei/H3C CLI command patterns (case-insensitive)
    CLI_PATTERNS = [
        r'^\s*router\s+',           # router ospf 1
        r'^\s*interface\s+',        # interface GigabitEthernet0/0
        r'^\s*ip\s+address\s+',     # ip address 10.0.0.1 255.255.255.0
        r'^\s*network\s+',          # network 10.0.0.0 0.0.0.255 area 0
        r'^\s*access-list\s+',      # access-list 10 permit ...
        r'^\s*do\s+show\s+',        # do show ip route
        r'^\s*exit\s*$',            # exit
        r'^\s*quit\s*$',            # quit (H3C/Huawei)
        r'^\s*no\s+shutdown\s*$',   # no shutdown
        r'^\s*switchport\s+',       # switchport mode access
        r'^\s*vlan\s+',             # vlan 10
        r'^\s*hostname\s+',         # hostname R1
        r'^\s*line\s+',             # line vty 0 4
        r'^\s*enable\s+',           # enable secret
        r'^\s*crypto\s+',           # crypto key generate
        r'^\s*spanning-tree\s+',    # spanning-tree mode
        r'^\s*router-id\s+',        # router-id 1.1.1.1
        r'^\s*area\s+',             # area 0
        r'^\s*redistribute\s+',     # redistribute connected
        r'^\s*default-information\s+', # default-information originate
        r'^\s*passive-interface\s+', # passive-interface default
        r'^\s*service-policy\s+',   # service-policy output POLICY
        r'^\s*class-map\s+',        # class-map match-all
        r'^\s*policy-map\s+',       # policy-map POLICY
        r'^\s*route-map\s+',        # route-map RMAP
        r'^\s*\[[\w\-]+\]',         # [SwitchA] or [SwitchA-ospf-1] (H3C/Huawei)
        r'^\s*<[\w\-]+>',           # <Switch> (H3C/Huawei)
    ]
    
    # Compile patterns for efficiency
    cli_regex = re.compile('|'.join(CLI_PATTERNS), re.IGNORECASE | re.MULTILINE)
    
    # Comment/filler patterns to remove
    COMMENT_PATTERNS = [
        r'^\s*#',                   # # This is a comment
        r'^\s*Note:',               # Note: Remember to...
        r'^\s*Important:',          # Important: Always check...
        r'^\s*Explanation:',        # Explanation: This command...
        r'^\s*Example:',            # Example: To configure...
        r'^\s*\*\*',                # ** Markdown bold markers
        r'^\s*---+\s*$',            # Horizontal rules
        r'^\s*Step\s+\d+:',         # Step 1: Configure...
        r'^\s*\d+\.',               # 1. First step...
    ]
    
    comment_regex = re.compile('|'.join(COMMENT_PATTERNS), re.IGNORECASE)
    
    filtered_chunks = []
    
    for chunk in procedural_chunks:
        if not chunk or not chunk.strip():
            continue
        
        lines = chunk.split('\n')
        
        # Heuristic 1: Count CLI command lines
        cli_line_count = sum(1 for line in lines if cli_regex.search(line))
        
        # Keep chunk only if it meets minimum CLI command threshold
        if cli_line_count < min_cli_commands:
            continue
        
        # Heuristic 2: Remove comment/filler lines while preserving structure
        refined_lines = []
        prev_was_empty = False
        
        for line in lines:
            # Skip excessive blank lines (max 1 consecutive)
            if not line.strip():
                if not prev_was_empty:
                    refined_lines.append(line)
                    prev_was_empty = True
                continue
            
            prev_was_empty = False
            
            # Skip comment/filler lines
            if comment_regex.match(line):
                continue
            
            # Keep the line (preserves indentation)
            refined_lines.append(line)
        
        # Only keep chunk if it still has substantial content after filtering
        if len(refined_lines) >= min_cli_commands:
            filtered_chunks.append('\n'.join(refined_lines).strip())
    
    # Stitch chunks together with clear separator
    if not filtered_chunks:
        return ""
    
    return '\n\n---\n\n'.join(filtered_chunks)


def extract_cli_heavy_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Extract and boost CLI-heavy chunks from a list of chunk dictionaries.
    
    Args:
        chunks: List of dicts with 'text' and optional 'score' keys
    
    Returns:
        List of chunks filtered to only CLI-heavy content with boosted scores
    """
    CLI_PATTERN = re.compile(
        r'^\s*(?:router|interface|ip\s+address|network|switchport|vlan|'
        r'access-list|exit|quit|no\s+shutdown|enable|crypto|spanning-tree|'
        r'line|hostname|area|redistribute|passive-interface|'
        r'\[[\w\-]+\]|<[\w\-]+>)',
        re.IGNORECASE | re.MULTILINE
    )
    
    cli_heavy = []
    for chunk in chunks:
        text = chunk.get("text", "")
        lines = text.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        
        if total_lines == 0:
            continue
        
        cli_lines = sum(1 for line in lines if CLI_PATTERN.search(line))
        cli_density = cli_lines / total_lines
        
        # Keep if CLI density >= 40%
        if cli_density >= 0.4:
            boosted_chunk = chunk.copy()
            # Boost score by CLI density
            if "score" in boosted_chunk:
                boosted_chunk["score"] = boosted_chunk["score"] * (1 + cli_density * 0.5)
            boosted_chunk["cli_density"] = cli_density
            cli_heavy.append(boosted_chunk)
    
    # Sort by score (highest first)
    cli_heavy.sort(key=lambda x: x.get("score", 0), reverse=True)
    return cli_heavy


def refine_code_chunks(code_chunks: List[str]) -> List[str]:
    """
    Apply aggressive filtering to code chunks to remove noise.
    Reinforces CLI-heavy content.
    
    Args:
        code_chunks: List of code chunk texts
    
    Returns:
        Filtered list of code chunks
    """
    return [
        chunk for chunk in [
            filter_and_refine_context([c], aggressive=True)
            for c in code_chunks
        ]
        if chunk  # Filter out empty results
    ]


# Example usage and testing
if __name__ == "__main__":
    # Mock procedural chunks from FAISS
    test_chunks = [
        """
# This is a comment explaining OSPF
Note: OSPF is a link-state routing protocol
router ospf 1
 router-id 1.1.1.1
 network 10.0.0.0 0.0.0.255 area 0
exit
        """,
        """
This chunk has no CLI commands.
It's just explanatory text.
**Important**: Always verify your configuration.
        """,
        """
interface GigabitEthernet0/0
 ip address 10.0.0.1 255.255.255.0
 no shutdown
exit
router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
        """,
        """
Example: To configure OSPF, use the following commands:
router ospf 1
        """
    ]
    
    print("=== ORIGINAL CHUNKS ===")
    for i, chunk in enumerate(test_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
    
    print("\n\n=== FILTERED CONTEXT ===")
    filtered = filter_and_refine_context(test_chunks)
    print(filtered)
    
    print("\n\n=== STATISTICS ===")
    print(f"Original chunks: {len(test_chunks)}")
    # Fix f-string backslash issue
    separator = '\n---\n'
    filtered_chunk_list = [c for c in filtered.split(separator) if c.strip()]
    print(f"Filtered chunks: {len(filtered_chunk_list)}")