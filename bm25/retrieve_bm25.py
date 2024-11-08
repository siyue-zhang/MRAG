import json
from pyserini.search.lucene import LuceneSearcher
import sys
sys.path.append('../')
from contriever.src.evaluation import SimpleTokenizer, has_answer
from utils import eval_recall, save_json_file, load_json_file
# import ipdb; ipdb.set_trace()

root = '/scratch/sz4651/Projects/metriever_final'
check_r = False
topk = 1000
num_examples = None
tokenizer = SimpleTokenizer()
output_path = f'{root}/TempRAGEval/BM25_output/TempRAGEval.json'
path = f'{root}/TempRAGEval/TempRAGEval.json'
sep='::'

if not check_r:
    with open(path, 'r') as file:
        examples = json.load(file)
    if num_examples:
        examples = examples[:num_examples]
    print('started.')
    for k, ex in enumerate(examples):
        # if k<489:
        #     continue
        # if k>360:
        #     break
        print(f'--{k}--')
        # dict_keys(['question', 'answers', 'ctxs'])
        # dict_keys(['id', 'title', 'text', 'score', 'hasanswer'])
        ctxs = []
        answers = ex['answers']
        question = ex['question']
        searcher = LuceneSearcher(f'{root}/bm25/index/enwiki-dec2021/indexed')
        hits = searcher.search(question, k=topk)
        for i in range(topk):
            id = hits[i].docid
            parts = id.split(sep)
            short_id = parts[0]
            title = sep.join(parts[1:])
            text = eval(searcher.doc(id).raw())['contents']
            h = has_answer(answers, text, tokenizer)
            ctx = {
                'id': short_id,
                'title': title,
                'text': text,
                'score': hits[i].score,
                'hasanswer': h
            }
            ctxs.append(ctx)
        ex['bm25_ctxs'] = ctxs
else:
    examples = load_json_file(output_path)

# print('\nContriever Performance')
# eval_recall(examples, 'ctxs', check_concat=False, include_title=True, verbose=False)
print('BM25 Performance')
print('Answer Recall')
eval_recall(examples, 'bm25_ctxs', ans_key='answers')
# print('Evidence Recall')
# eval_recall(examples, 'bm25_ctxs', ans_key='gold_evidences')


if not check_r:
    save_json_file(output_path, examples)

