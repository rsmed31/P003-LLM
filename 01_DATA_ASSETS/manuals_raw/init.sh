rm -f ./docs/raw_chunks.json

python PDFExtractorV2.py
python ChunkCreationV4.py
python EmbeddingCreation.py
python ../postgres_api/test.py