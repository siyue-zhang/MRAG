
from utils import load_json_file
from ipdb import set_trace
wiki = "/scratch/sz4651/Projects/modular_retriever/enwiki-dec2021/psgs_w100.json"
wiki = load_json_file(wiki)
print(wiki["Governor of the Virgin Islands"])

set_trace()