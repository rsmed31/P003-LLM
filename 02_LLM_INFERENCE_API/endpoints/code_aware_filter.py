import re


def filter_and_refine_context(procedural_chunks: list[str]) -> str:
    """
    Filters and refines procedural context chunks using code-aware heuristics.
    
    Args:
        procedural_chunks: List of raw documentation/example chunks from FAISS
    
    Returns:
        Single string containing filtered, high-quality CLI context ready for RAG
    
    Filtering Strategy:
        - Heuristic 1: Keep chunks with ≥2 executable CLI command lines
        - Heuristic 2: Remove non-CLI explanatory text and comments
        - Stitch remaining chunks with clear separators
    """
    
    # Cisco CLI command patterns (case-insensitive)
    CLI_PATTERNS = [
        r'^\s*router\s+',           # router ospf 1
        r'^\s*interface\s+',        # interface GigabitEthernet0/0
        r'^\s*ip\s+address\s+',     # ip address 10.0.0.1 255.255.255.0
        r'^\s*network\s+',          # network 10.0.0.0 0.0.0.255 area 0
        r'^\s*access-list\s+',      # access-list 10 permit ...
        r'^\s*do\s+show\s+',        # do show ip route
        r'^\s*exit\s*$',            # exit
        r'^\s*no\s+shutdown\s*$',   # no shutdown
        r'^\s*switchport\s+',       # switchport mode access
        r'^\s*vlan\s+',             # vlan 10
        r'^\s*hostname\s+',         # hostname R1
        r'^\s*line\s+',             # line vty 0 4
        r'^\s*enable\s+',           # enable secret
        r'^\s*crypto\s+',           # crypto key generate
        r'^\s*spanning-tree\s+',    # spanning-tree mode
        r'^\s*router-id\s+',        # router-id 1.1.1.1
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
    ]
    
    comment_regex = re.compile('|'.join(COMMENT_PATTERNS), re.IGNORECASE)
    
    filtered_chunks = []
    
    for chunk in procedural_chunks:
        if not chunk or not chunk.strip():
            continue
        
        lines = chunk.split('\n')
        
        # Heuristic 1: Count CLI command lines
        cli_line_count = sum(1 for line in lines if cli_regex.search(line))
        
        # Keep chunk only if it has at least 2 CLI commands
        if cli_line_count < 2:
            continue
        
        # Heuristic 2: Remove comment/filler lines
        refined_lines = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip comment/filler lines
            if comment_regex.match(line):
                continue
            
            # Keep the line
            refined_lines.append(line)
        
        # Only keep chunk if it still has content after filtering
        if refined_lines:
            filtered_chunks.append('\n'.join(refined_lines))
    
    # Stitch chunks together with clear separator
    if not filtered_chunks:
        return ""
    
    return '\n---\n'.join(filtered_chunks)


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
    print(f"Filtered chunks: {len([c for c in filtered.split('\\n---\\n') if c.strip()])}")
