
from utils import load_json_file, eval_recall

examples = load_json_file('/scratch/sz4651/Projects/metriever_final/retrieved/situatedqa_contriever_metriever_minilm12_llama_70b_qfs5_outputs.json')


def separate_samples(examples):
# separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = [], [], []
    for example in examples:
        if example['time_relation'] == '':
            examples_notime.append(example)
        elif int(example['exact']) == 1:
            examples_exact.append(example)
        else:
            examples_not_exact.append(example)
    return examples_notime, examples_exact, examples_not_exact


examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

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