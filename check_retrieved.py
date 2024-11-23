import json
from utils import *
from metriever import separate_samples


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


examples = load_json_file('./retrieved/situatedqa_contriever_metriever_bgegemma_llama_8b_qfs5_outputs.json')
examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

# for i,ex in enumerate(examples):
#     print(f'\n---{i}---')
#     question = ex['question']
#     answer = ex['answers']
#     print(question)
#     print(answer,'\n')
#     print_top_ctx(ex, 10, 'snt_hybrid_rank')


# print('\n**** Answers ****\n')
# print('~~~~~~~ctx_keyword_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='ctx_keyword_rank', ans_key='answers')
# print('~~~~~~~ctx_semantic_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='ctx_semantic_rank', ans_key='answers')
# print('~~~~~~~snt_keyword_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='snt_keyword_rank', ans_key='answers')
# print('~~~~~~~snt_hybrid_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='snt_hybrid_rank', ans_key='answers') 

# print('\n**** Gold Evidences ****\n')
# print('~~~~~~~ctx_keyword_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='ctx_keyword_rank', ans_key='gold_evidences')
# print('~~~~~~~ctx_semantic_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='ctx_semantic_rank', ans_key='gold_evidences')
# print('~~~~~~~snt_keyword_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='snt_keyword_rank', ans_key='gold_evidences')
# print('~~~~~~~snt_hybrid_rank~~~~~~~~')
# eval_recall(examples_exact, ctxs_key='snt_hybrid_rank', ans_key='gold_evidences') 

print('=====================================')

print('\n**** Answers ****\n')
print('~~~~~~~ctx_keyword_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='ctx_keyword_rank', ans_key='answers')
print('~~~~~~~ctx_semantic_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='ctx_semantic_rank', ans_key='answers')
print('~~~~~~~snt_keyword_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='snt_keyword_rank', ans_key='answers')
print('~~~~~~~snt_hybrid_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='snt_hybrid_rank', ans_key='answers') 

print('\n**** Gold Evidences ****\n')
print('~~~~~~~ctx_keyword_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='ctx_keyword_rank', ans_key='gold_evidences')
print('~~~~~~~ctx_semantic_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='ctx_semantic_rank', ans_key='gold_evidences')
print('~~~~~~~snt_keyword_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='snt_keyword_rank', ans_key='gold_evidences')
print('~~~~~~~snt_hybrid_rank~~~~~~~~')
eval_recall(examples_not_exact, ctxs_key='snt_hybrid_rank', ans_key='gold_evidences') 



