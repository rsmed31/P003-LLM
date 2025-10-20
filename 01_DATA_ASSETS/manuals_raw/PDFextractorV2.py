#!/usr/bin/env python3
"""
PDF/TXT -> bereinigte, vereinheitlichte Textdateien für OSPF und VLAN

Features:
- Prozessor für zwei relative Ordner: ./docs/ospf/ und ./docs/vlan/
- Unterstützt PDF (PyMuPDF / fitz) und TXT
- "Mittel" Cleaning (Option B):
  * Entfernt wiederkehrende Header/Footer (automatisch erkannt)
  * Entfernt Seitenzahlen wie "Page X of Y" oder "X/Y"
  * Entfernt Trennlinien (---, ___ etc.)
  * Fügt an Bindestrich getrennte Wörter zusammen (z.B. "confi-\nguration" -> "configuration")
  * Entfernt überflüssige Leerzeilen und vereinfacht Whitespace
- Schreibt je Ordner eine Datei: ospf_manual.txt / vlan_manual.txt
- Nutzt doclist.txt (Liste verarbeiteter Dateien, relativ), um doppelte Verarbeitung zu vermeiden
- Gut erweiterbar: danach kannst du die TXT-Dateien für Chunking / Embeddings / FAISS weiterverwenden
"""

import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import logging

# --- Konfigurationen / Defaults ---
DEFAULT_OSPF_DIR = Path("./docs/ospf")
DEFAULT_VLAN_DIR = Path("./docs/vlan")
DOCLIST_PATH = Path("doclist.txt")
OUTPUT_OSPF = Path("ospf_manual.txt")
OUTPUT_VLAN = Path("vlan_manual.txt")
# Minimum fraction of pages a header/footer must appear on to be considered repeating
HEADER_FOOTER_THRESHOLD = 0.5

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------
# Hilfsfunktionen
# -------------------------
def list_documents(folder: Path) -> List[Path]:
    """Gibt alle .pdf und .txt Dateien in einem Ordner zurück (nicht rekursiv)."""
    if not folder.exists():
        logging.warning("Ordner %s existiert nicht.", folder)
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt")]
    files.sort()
    return files


def read_txt_file(path: Path) -> str:
    """Liest eine TXT-Datei."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_pdf(path: Path) -> List[str]:
    """
    Extrahiert Text pro Seite aus einem PDF (fitz).
    Gibt eine Liste mit den Page-Texten zurück.
    """
    texts = []
    doc = fitz.open(str(path))
    for page in doc:
        try:
            text = page.get_text()
        except Exception as e:
            logging.warning("Fehler beim Lesen der Seite %d von %s: %s", page.number, path, e)
            text = ""
        texts.append(text if text is not None else "")
    doc.close()
    return texts


# -------------------------
# Cleaning: Option B (mittel)
# -------------------------
def detect_repeating_headers_footers(page_texts: List[str], head_lines=2, tail_lines=2) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Scannt die ersten/letzten 'head_lines'/'tail_lines' jeder Seite und zählt die Häufigkeit.
    Gibt zwei Dicts zurück: header_counts, footer_counts
    """
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
        # header candidate: first head_lines joined
        if len(lines) >= 1:
            header_candidate = " | ".join(lines[:head_lines]) if len(lines) >= head_lines else lines[0]
            header_counts[header_candidate] = header_counts.get(header_candidate, 0) + 1
        # footer candidate: last tail_lines joined
        if len(lines) >= 1:
            footer_candidate = " | ".join(lines[-tail_lines:]) if len(lines) >= tail_lines else lines[-1]
            footer_counts[footer_candidate] = footer_counts.get(footer_candidate, 0) + 1

    return header_counts, footer_counts


def find_common_candidates(counts: Dict[str, int], page_count: int, threshold_frac: float) -> List[str]:
    """Gibt Kandidaten zurück, die in mehr als threshold_frac * page_count Seiten vorkommen."""
    threshold = max(1, int(page_count * threshold_frac))
    return [k for k, v in counts.items() if v >= threshold]


def remove_header_footer_from_page(text: str, headers: List[str], footers: List[str]) -> str:
    """
    Entfernt erkannte Header/Footers (genauer: Zeilen, die gleich sind).
    Falls exakte Übereinstimmung in den ersten/letzten Zeilen vorhanden ist, werden sie entfernt.
    """
    if not text.strip():
        return text
    lines = text.splitlines()
    # remove leading matching headers
    for h in headers:
        h_parts = [part.strip() for part in h.split("|")]
        if len(lines) >= len(h_parts):
            candidate = " | ".join([ln.strip() for ln in lines[:len(h_parts)] if ln.strip() != ""])
            if candidate == h:
                # remove these lines
                lines = lines[len(h_parts):]
                break
    # remove trailing matching footers
    for f in footers:
        f_parts = [part.strip() for part in f.split("|")]
        if len(lines) >= len(f_parts):
            candidate = " | ".join([ln.strip() for ln in lines[-len(f_parts):] if ln.strip() != ""])
            if candidate == f:
                lines = lines[:-len(f_parts)]
                break
    return "\n".join(lines)


def remove_page_numbers(text: str) -> str:
    """Entfernt typische Seitenzahl-Formate."""
    # Beispiele: "Page 1 of 10", "Page 1", "1/10", "1 - 10" in eigenen Zeilen
    patterns = [
        r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$",   # Page 1 of 10
        r"^\s*\d+\s*/\s*\d+\s*$",             # 1/10
        r"^\s*\d+\s*-\s*\d+\s*$",             # 1 - 10
        r"^\s*seite\s*\d+\s*(von\s*\d+)?\s*$" # deutsche "Seite"
    ]
    lines = text.splitlines()
    new_lines = []
    for ln in lines:
        stripped = ln.strip().lower()
        if any(re.match(pat, stripped) for pat in patterns):
            continue
        new_lines.append(ln)
    return "\n".join(new_lines)


def medium_clean_page(text: str, headers: List[str], footers: List[str]) -> str:
    """
    Führt das "mittlere" Cleaning für eine einzelne Seite aus.
    - Entfernt identifizierte Header/Footers
    - Entfernt Seitenzahlen
    - Entfernt Trennlinien
    - Fügt an Bindestrich getrennte Wörter zusammen
    - Normalisiert Whitespace (mehrfache Leerzeilen -> max 2)
    - Ersetzt einzelne Zeilenumbrüche innerhalb von Absätzen mit Leerzeichen, behält doppelte Umbrüche als Absatztrenner
    """
    if not text:
        return ""

    # 1) Entferne Header/Footer (falls erkannt)
    text = remove_header_footer_from_page(text, headers, footers)

    # 2) Entferne Seitenzahlen
    text = remove_page_numbers(text)

    # 3) Entferne häufige Trennlinien (---, ___, ===)
    text = re.sub(r"^[\-\=_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 4) Kombiniere an Bindestrich getrennte Wörter (z.B. "confi-\n guration" oder "confi-\nconfiguration")
    text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)

    # 5) Entferne harte Zeilenumbrüche innerhalb von Absätzen:
    #    - Wenn zwischen zwei Zeilen kein leerer Zeilenumbruch ist, verbinde mit Space.
    #    - Doppelter Leerzeilen bleibt erhalten (Paragraphseparator).
    lines = text.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            # preserve paragraph break
            out_lines.append("")
            i += 1
            continue
        # collect consecutive non-empty lines into one paragraph line
        paragraph_lines = []
        while i < len(lines) and lines[i].strip() != "":
            paragraph_lines.append(lines[i].strip())
            i += 1
        # join with single space
        joined = " ".join(paragraph_lines)
        out_lines.append(joined)
    cleaned = "\n".join(out_lines)

    # 6) Normalize multiple blank lines -> max 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7) Normalize spaces
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # Trim
    cleaned = cleaned.strip() + "\n"
    return cleaned


# -------------------------
# Haupt-Workflow
# -------------------------
def process_folder(folder: Path, output_file: Path, doclist_path: Path):
    """
    Verarbeitet alle PDF/TXT Dateien in 'folder', erzeugt ein einheitliches Output-File.
    """
    logging.info("Starte Verarbeitung für Ordner: %s", folder)
    docs = list_documents(folder)
    if not docs:
        logging.info("Keine Dokumente in %s gefunden.", folder)
        return

    # Lade doclist
    processed = set()
    if doclist_path.exists():
        with doclist_path.open("r", encoding="utf-8") as f:
            processed = set([line.strip() for line in f if line.strip()])

    # Filter out already processed files
    to_process = [p for p in docs if str(p) not in processed and p.name not in processed]
    if not to_process:
        logging.info("Keine neuen Dokumente in %s (alle bereits verarbeitet).", folder)
        return

    # 1) Sammle texts per file
    all_cleaned_pages: List[str] = []  # for header/footer detection we need page-level texts
    file_page_map: List[Tuple[Path, List[str]]] = []

    for p in to_process:
        logging.info("Lesen: %s", p)
        if p.suffix.lower() == ".txt":
            raw = read_txt_file(p)
            # treat whole text as single "page" to keep logic consistent
            page_texts = [raw]
        else:
            page_texts = extract_text_from_pdf(p)

        file_page_map.append((p, page_texts))
        all_cleaned_pages.extend(page_texts)

    # 2) Detect repeating headers/footers across pages in the set (per-folder)
    header_counts, footer_counts = detect_repeating_headers_footers(all_cleaned_pages, head_lines=2, tail_lines=2)
    page_count = max(1, len(all_cleaned_pages))
    common_headers = find_common_candidates(header_counts, page_count, HEADER_FOOTER_THRESHOLD)
    common_footers = find_common_candidates(footer_counts, page_count, HEADER_FOOTER_THRESHOLD)

    logging.info("Erkannte wiederkehrende Header (falls any): %s", common_headers)
    logging.info("Erkannte wiederkehrende Footer (falls any): %s", common_footers)

    # 3) Clean pages and aggregate into one big text for the folder
    aggregated_parts: List[str] = []
    for p, page_texts in file_page_map:
        aggregated_parts.append(f"\n\n=== START OF FILE: {p.name} ===\n\n")
        for page_idx, page_text in enumerate(page_texts):
            cleaned = medium_clean_page(page_text, common_headers, common_footers)
            # optional: add small marker between pages to not lose boundaries
            aggregated_parts.append(cleaned + "\n")
        aggregated_parts.append(f"\n\n=== END OF FILE: {p.name} ===\n\n")

    aggregated_text = "\n".join(aggregated_parts).strip() + "\n"

    # 4) Final pass: remove remaining artifact lines (very short weird lines), but be conservative:
    #    remove lines that contain only non-word punctuation or sequences like "----"
    filtered_lines = []
    for ln in aggregated_text.splitlines():
        if re.match(r"^[\W_]{3,}$", ln.strip()):  # lines of only punctuation/underscores/dashes
            continue
        # keep other lines (including short meaningful lines like "Introduction")
        filtered_lines.append(ln.rstrip())
    final_text = "\n".join(filtered_lines).strip() + "\n"

    # 5) Schreibe die Output-Datei (überschreibt vorhandene Ausgabe für dieses Thema)
    with output_file.open("w", encoding="utf-8") as out_f:
        out_f.write(final_text)
    logging.info("Schrieb bereinigte Datei: %s (Bytes: %d)", output_file, output_file.stat().st_size)

    # 6) Markiere die verarbeiteten Dateien in doclist (relative Pfade)
    with doclist_path.open("a", encoding="utf-8") as dl:
        for p, _ in file_page_map:
            rel = str(p)
            dl.write(rel + "\n")
    logging.info("Markierte %d Dateien in %s", len(file_page_map), doclist_path)


def main(args):
    ospf_dir = Path(args.ospf_dir)
    vlan_dir = Path(args.vlan_dir)
    doclist = Path(args.doclist)
    out_ospf = Path(args.out_ospf)
    out_vlan = Path(args.out_vlan)

    process_folder(ospf_dir, out_ospf, doclist)
    process_folder(vlan_dir, out_vlan, doclist)
    logging.info("Fertig. Outputs: %s , %s", out_ospf, out_vlan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrahiere & bereinige Text aus PDF/TXT in ./docs/ospf und ./docs/vlan")
    parser.add_argument("--ospf-dir", type=str, default=str(DEFAULT_OSPF_DIR), help="Ordner mit OSPF Dokumenten (relativ)")
    parser.add_argument("--vlan-dir", type=str, default=str(DEFAULT_VLAN_DIR), help="Ordner mit VLAN Dokumenten (relativ)")
    parser.add_argument("--doclist", type=str, default=str(DOCLIST_PATH), help="Datei, die verarbeitete Dokumente speichert")
    parser.add_argument("--out-ospf", type=str, default=str(OUTPUT_OSPF), help="Output-Datei für OSPF (wird überschrieben)")
    parser.add_argument("--out-vlan", type=str, default=str(OUTPUT_VLAN), help="Output-Datei für VLAN (wird überschrieben)")
    parsed = parser.parse_args()
    main(parsed)
