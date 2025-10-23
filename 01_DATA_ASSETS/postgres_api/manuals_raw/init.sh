rm -f ./docs/raw_chunks.json

python3 ./PDFExtractorV2.py
python3 ./ChunkCreationV4.py
python3 ./EmbeddingCreation.py
python3 ../postgres_api/test.py