import os
import json

# Text Cleaning
def clean_text(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()

# Chunking
def chunk_text(text: str, chunk_size, overlap):
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            split_index = text.rfind(' ', start, end)
            if split_index == -1 or split_index <= start:
                split_index = end
        else:
            split_index = text_length

        chunk = text[start:split_index].strip()
        if chunk:
            yield chunk

        new_start = split_index - overlap
        if new_start <= start:
            new_start = split_index
        start = new_start

# Streamed JSON Writing
def process_directory_json(input_dir,
                           output_file,
                           chunk_size,
                           overlap,
                           log_every):
    
    total_chunks = 0
    first = True

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("[\n")

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(input_dir, filename)
                print(f"Processing {filename}...")

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception as e:
                    print(f"Fehler beim Lesen von {filename}: {e}")
                    continue

                cleaned = clean_text(text)

                for i, c in enumerate(chunk_text(cleaned, chunk_size, overlap)):
                    item = {
                        "source": filename,
                        "chunk_index": i,
                        "text": c
                    }

                    if not first:
                        out.write(",\n")
                    json.dump(item, out, ensure_ascii=False)
                    first = False

                    total_chunks += 1
                    if total_chunks % log_every == 0:
                        print(f"{total_chunks} Chunks bisher gespeichert...")

        out.write("\n]\n")

    print(f"✅ Fertig! {total_chunks} Chunks gespeichert in {output_file}")

# Main
if __name__ == "__main__":
    input_dir = "./docs/"
    output_file = "./docs/raw_chunks.json"
    chunk_size = 1000
    overlap = 100
    process_directory_json(input_dir, output_file, chunk_size, overlap,50)
