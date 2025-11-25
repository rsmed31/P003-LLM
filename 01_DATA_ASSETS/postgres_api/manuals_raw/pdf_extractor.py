
import os
import re
import fitz
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import logging

DEFAULT_OSPF_DIR = Path("./manuals_raw/docs/ospf")
DEFAULT_VLAN_DIR = Path("./manuals_raw/docs/vlan")
DOCLIST_PATH = Path("./manuals_raw/docs/doclist.txt")
OUTPUT_OSPF = Path("./manuals_raw/docs/ospf_manual.txt")
OUTPUT_VLAN = Path("./manuals_raw/docs/vlan_manual.txt")
HEADER_FOOTER_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def list_documents(folder: Path) -> List[Path]:
    if not folder.exists():
        logging.warning("Folder %s does not exist.", folder)
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")])

def read_txt_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_pdf(path: Path) -> List[str]:
    texts = []
    doc = fitz.open(str(path))
    for page in doc:
        try:
            texts.append(page.get_text() or "")
        except Exception as e:
            logging.warning("Error reading page %d of %s: %s", page.number, path, e)
            texts.append("")
    doc.close()
    return texts

def detect_repeating_headers_footers(page_texts: List[str], head_lines=2, tail_lines=2) -> Tuple[Dict[str, int], Dict[str, int]]:
    header_counts, footer_counts = {}, {}
    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            header_candidate = " | ".join(lines[:head_lines]) if len(lines) >= head_lines else lines[0]
            footer_candidate = " | ".join(lines[-tail_lines:]) if len(lines) >= tail_lines else lines[-1]
            header_counts[header_candidate] = header_counts.get(header_candidate, 0) + 1
            footer_counts[footer_candidate] = footer_counts.get(footer_candidate, 0) + 1
    return header_counts, footer_counts

def find_common_candidates(counts: Dict[str, int], page_count: int, threshold_frac: float) -> List[str]:
    threshold = max(1, int(page_count * threshold_frac))
    return [k for k, v in counts.items() if v >= threshold]

def remove_header_footer_from_page(text: str, headers: List[str], footers: List[str]) -> str:
    if not text.strip():
        return text
    lines = text.splitlines()
    for h in headers:
        h_parts = [p.strip() for p in h.split("|")]
        if len(lines) >= len(h_parts) and " | ".join([ln.strip() for ln in lines[:len(h_parts)] if ln.strip()]) == h:
            lines = lines[len(h_parts):]
            break
    for f in footers:
        f_parts = [p.strip() for p in f.split("|")]
        if len(lines) >= len(f_parts) and " | ".join([ln.strip() for ln in lines[-len(f_parts):] if ln.strip()]) == f:
            lines = lines[:-len(f_parts)]
            break
    return "\n".join(lines)

def remove_page_numbers(text: str) -> str:
    patterns = [
        r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$",
        r"^\s*\d+\s*/\s*\d+\s*$",
        r"^\s*\d+\s*-\s*\d+\s*$",
        r"^\s*seite\s*\d+\s*(von\s*\d+)?\s*$"
    ]
    return "\n".join(ln for ln in text.splitlines() if not any(re.match(pat, ln.strip().lower()) for pat in patterns))

def medium_clean_page(text: str, headers: List[str], footers: List[str]) -> str:
    if not text:
        return ""
    text = remove_header_footer_from_page(text, headers, footers)
    text = remove_page_numbers(text)
    text = re.sub(r"^[\-\=_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
    lines = text.splitlines()
    out_lines, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            out_lines.append("")
            i += 1
            continue
        paragraph = []
        while i < len(lines) and lines[i].strip():
            paragraph.append(lines[i].strip())
            i += 1
        out_lines.append(" ".join(paragraph))
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip() + "\n"

def process_folder(folder: Path, output_file: Path, doclist_path: Path):
    logging.info("Processing folder: %s", folder)
    docs = list_documents(folder)
    if not docs:
        logging.info("No documents found in %s", folder)
        return
    processed = set()
    if doclist_path.exists():
        with doclist_path.open("r", encoding="utf-8") as f:
            processed = set(line.strip() for line in f if line.strip())
    to_process = [p for p in docs if str(p) not in processed and p.name not in processed]
    if not to_process:
        logging.info("No new documents in %s", folder)
        return
    all_pages, file_page_map = [], []
    for p in to_process:
        logging.info("Reading: %s", p)
        page_texts = [read_txt_file(p)] if p.suffix.lower() == ".txt" else extract_text_from_pdf(p)
        file_page_map.append((p, page_texts))
        all_pages.extend(page_texts)
    headers, footers = detect_repeating_headers_footers(all_pages)
    page_count = max(1, len(all_pages))
    common_headers = find_common_candidates(headers, page_count, HEADER_FOOTER_THRESHOLD)
    common_footers = find_common_candidates(footers, page_count, HEADER_FOOTER_THRESHOLD)
    logging.info("Detected headers: %s", common_headers)
    logging.info("Detected footers: %s", common_footers)
    aggregated = []
    for p, pages in file_page_map:
        for page in pages:
            aggregated.append(medium_clean_page(page, common_headers, common_footers) + "\n")
    final_text = "\n".join(ln.rstrip() for ln in "\n".join(aggregated).splitlines() if not re.match(r"^[\W_]{3,}$", ln.strip()))
    with output_file.open("w", encoding="utf-8") as f:
        f.write(final_text)
    logging.info("Wrote cleaned file: %s (Bytes: %d)", output_file, output_file.stat().st_size)
    with doclist_path.open("a", encoding="utf-8") as dl:
        for p, _ in file_page_map:
            dl.write(str(p) + "\n")
    logging.info("Marked %d files in %s", len(file_page_map), doclist_path)

def main(args):
    process_folder(Path(args.ospf_dir), Path(args.out_ospf), Path(args.doclist))
    process_folder(Path(args.vlan_dir), Path(args.out_vlan), Path(args.doclist))
    logging.info("Done. Outputs: %s , %s", args.out_ospf, args.out_vlan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract & clean text from TXT/PDF in ./docs/ospf and ./docs/vlan")
    parser.add_argument("--ospf-dir", type=str, default=str(DEFAULT_OSPF_DIR))
    parser.add_argument("--vlan-dir", type=str, default=str(DEFAULT_VLAN_DIR))
    parser.add_argument("--doclist", type=str, default=str(DOCLIST_PATH))
    parser.add_argument("--out-ospf", type=str, default=str(OUTPUT_OSPF))
    parser.add_argument("--out-vlan", type=str, default=str(OUTPUT_VLAN))
    main(parser.parse_args())
