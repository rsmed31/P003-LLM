import os
import json

# --------------------------
# 1) Clean Text (sehr basic)
# --------------------------
def clean_text(text: str) -> str:
    """
    Entfernt überflüssige Whitespaces und normalisiert Zeilenumbrüche.
    (Hier sehr einfach gehalten – kann später erweitert werden.)
    """
    text = text.replace('\r\n', '\n')  # Windows zu Unix
    text = text.replace('\r', '\n')
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()

# ------------------------------------------
# 2) Recursive Character-Splitting (Basic)
# ------------------------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Zerlegt Text in fixe Größen.
    Wenn möglich an Leerzeichen splitten (sanfter).
    overlap = wieviel vom Ende des vorherigen Chunks wiederholt wird.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        # Versuch, am letzten Leerzeichen zu trennen
        if end < text_length:
            split_index = text.rfind(' ', start, end)
            if split_index == -1 or split_index <= start:
                split_index = end  # notfalls hart trennen
        else:
            split_index = text_length

        chunk = text[start:split_index].strip()
        if chunk:
            chunks.append(chunk)

        # Overlap einbauen
        start = split_index - overlap
        if start < 0:
            start = 0

        # Wenn kein Fortschritt -> infinite loop vermeiden
        if start >= split_index:
            start = split_index

    return chunks

# ------------------------------------------
# 3) Main - Textdateien einlesen & chunken
# ------------------------------------------
def process_directory(input_dir: str, output_file: str = "raw_chunks.json", chunk_size: int = 1000, overlap: int = 200):
    all_chunks = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            cleaned = clean_text(text)
            chunks = chunk_text(cleaned, chunk_size, overlap)

            # Optional: Chunk-ID mit Dateiname
            for i, c in enumerate(chunks):
                all_chunks.append({
                    "source": filename,
                    "chunk_index": i,
                    "text": c
                })

    # Speichern als JSON-Liste
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_chunks, out, ensure_ascii=False, indent=2)

    print(f"Fertig! {len(all_chunks)} Chunks gespeichert in {output_file}")


# ------------------------------------------
# 4) Ausführen
# ------------------------------------------
if __name__ == "__main__":
    input_dir = "./"
    process_directory(input_dir)