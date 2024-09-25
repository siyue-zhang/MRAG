import csv
import json
from tqdm import tqdm

# {
#   "id": "doc1",
#   "contents": "this is the contents."
# }


wiki_json = '/scratch/sz4651/Projects/porqa/wikipedia/enwiki-dec2021/psgs_w100.json'
with open(wiki_json, 'r', encoding='utf-8') as file:
    data = json.load(file)

to_save = []
for page in data:
    print(page)
    for ctx in data[page]:
        to_save.append({
            "id": ctx['id']+'::'+ctx['title'],
            "contents": ctx['text']
        })

with open("/scratch/sz4651/Projects/modular_retriever/bm25/index/enwiki-dec2021/input/documents.json", 'w') as json_file:
    json.dump(to_save, json_file, indent=4)

print(f"Data has been saved")


# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input /scratch/sz4651/Projects/porqa/bm25/index/enwiki-dec2021/ \
#   --index /scratch/sz4651/Projects/porqa/bm25/index/enwiki-dec2021/indexed/ \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 1 \
#   --storePositions --storeDocvectors --storeRaw