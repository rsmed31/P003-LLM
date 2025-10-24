rm ./docs/raw_chunks.json

python PDFExtractorV2.py
python ChunkCreationV4.py
python EmbeddingCreation.py
python ../Database/test.py #TODO Maybe remove test later