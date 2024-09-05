import json
from utils import *
from metriever import separate_samples

def load_json_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def print_top_ctx(ex, topk, ctx_key):
    for i, ctx in enumerate(ex[ctx_key]):
        text = ctx['title'] + ' ' + ctx['text']
        answer = ex['answers']
        print(f'== {i} ==')
        print(text)
        print('hasanswer: ', has_answer(answer, text, tokenizer))
        if 'QFS_summary' in ctx:
            print('QFS: ', ctx['QFS_summary'])
        print('\n')
        if i == topk:
            break


examples = load_json_file('./retrieved/contriever_metriever_minilm12_llama_8b_outputs.json')
examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

for i,ex in enumerate(examples):
    print(f'\n---{i}---')
    question = ex['question']
    answer = ex['answers']
    print(question)
    print(answer,'\n')
    print_top_ctx(ex, 10, 'snt_hybrid_rank')



