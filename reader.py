from utils import *
from prompts import *
from metriever import separate_samples

import pandas as pd
# import ipdb; ipdb.set_trace()
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams
 
from temp_eval import normalize

def reader_pipeline(llm, prompts):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    responses = [res.split('Question:')[0] for res in responses]
    responses = [res.replace('\n','').strip() for res in responses]
    return responses 

def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--retriever-output', type=str, default="bm25_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=10)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='bm25', #
    )
    parser.add_argument('--ctx-key-s2', type=str, default=None)

    args = parser.parse_args()
    args.l = llm_names(args.llm)
    args.llm_name = deepcopy(args.llm)

    if args.stage1_model=='bm25':
        args.ctx_key_s1 = 'bm25_ctxs'
    elif args.stage1_model=='hybrid':
        args.ctx_key_s1 = 'hybrid_ctxs'
    else:
        args.ctx_key_s1 = 'ctxs'

    # load llm 
    flg = '70b' in args.llm_name
    if flg:
        args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ")
    else:
        args.llm = LLM(args.l, tensor_parallel_size=1, dtype='half', max_model_len=4096)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]
    
    # only keep situatedqa and timeqa samples for this code
    # examples = [ex for ex in examples if ex["source"] != 'dbpedia']
    # examples =  examples[-700:]

    ########  QA  ######## 
    if args.param_pred:
        prompts = [zc_prompt(ex['question']) for ex in examples]
        param_preds = reader_pipeline(args.llm, prompts)
        print('zero context prediction finished.')

    prompts = []
    texts = []
    tmp_key = args.ctx_key_s2 if args.ctx_key_s2 else args.ctx_key_s1
    for ex in examples:
        text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
        texts.append(text)
        prompt = c_prompt(ex['question'], text)
        prompts.append(prompt)
    rag_preds = reader_pipeline(args.llm, prompts)
    print(f'{tmp_key} top {args.ctx_topk} contexts prediction finished.')

    to_save=[]
    for k, ex in enumerate(examples):
        question = ex['question']
        gold_evidences = ex['gold_evidences']

        # annotate each ctx if it contains answer and gold evidence
        for ctx in ex[args.ctx_key_s1]:
            ctx['hasanswer'] = str(has_answer(ex['answers'], ctx['title']+' '+ctx['text'], tokenizer))
        try:
            ans_index = [ctx['hasanswer'] for ctx in ex[args.ctx_key_s1]].index('True')+1
        except ValueError:
            ans_index = -1
        for ctx in ex[args.ctx_key_s1]:
            ctx['hasgold'] = str(has_answer(gold_evidences, ctx['title']+' '+ctx['text'], tokenizer))
        try:
            gold_index = [ctx['hasgold'] for ctx in ex[args.ctx_key_s1]].index('True')+1
        except ValueError:
            gold_index = -1

        if args.ctx_key_s2:
            for ctx in ex[args.ctx_key_s2]:
                ctx['hasanswer'] = str(has_answer(ex['answers'], ctx['title']+' '+ctx['text'], tokenizer))
            try:
                ans_index_reranker = [ctx['hasanswer'] for ctx in ex[args.ctx_key_s2]].index('True')+1
            except ValueError:
                ans_index_reranker = -1
            for ctx in ex[args.ctx_key_s2]:
                ctx['hasgold'] = str(has_answer(gold_evidences, ctx['title']+' '+ctx['text'], tokenizer))
            try:
                gold_index_reranker = [ctx['hasgold'] for ctx in ex[args.ctx_key_s2]].index('True')+1
            except ValueError:
                gold_index_reranker = -1

        rag_pred = rag_preds[k]
        ex['rag_pred'] = rag_pred
        ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])

        for item in ['QFS_summary']:
            if 'QFS_summary' not in ex[args.ctx_key][0]:
                for ctx in ex[args.ctx_key]:
                    ctx[item]=''

        s1_ctx_text = '\n\n'.join([f" {t+1} | {ctx['hasanswer']} | {ctx['title']} | {ctx['text']}" for t, ctx in enumerate(ex[args.ctx_key_s1][:20])])
        s2_ctx_text = '\n\n'.join([f"{t+1} | {ctx['hasanswer']} | {ctx['title']} | {ctx['text']}\nQFS: {ctx['QFS_summary']}" for  t, ctx in enumerate(ex[args.ctx_key_s2][:20])]) if args.ctx_key_s2 else ''
        
        result = {
            'id': ex['id'],
            'source': ex['source'],
            'question': question,
            'exact': ex['exact'],
            'answers': ex['answers'],
            'time_relation': ex['time_relation'],
            'contriever_ans_hit': ans_index,
            'contriever_gold_hit': gold_index,
            'contriever_ctxs': s1_ctx_text,
            'reranker_ans_hit': ans_index_reranker,
            'reranker_gold_hit': gold_index_reranker,
            'reranker_ctxs': s2_ctx_text,
            'rag_pred': ex['rag_pred'],
            'rag_acc': ex['rag_acc'],
            'gold_evidence_1': gold_evidences[0] if len(gold_evidences)>0 else '',
            'gold_evidence_2': gold_evidences[1] if len(gold_evidences)>1 else '',
            'top_snts': ex['top_snts'] if 'top_snts' in ex else '',
            }

        if args.param_pred:
            param_pred = param_preds[k]
            ex['param_pred'] = param_pred
            ex['param_acc'] = int(normalize(param_pred) in [normalize(ans) for ans in ex['answers']])
            result.update({
                'param_pred': ex['param_pred'],
                'param_acc': ex['param_acc'],
            })

        to_save.append(result)

    to_save_df = pd.DataFrame(to_save)
    retriever_name = args.retriever_output.split('/')[-1].split('_outputs')[0]
    result_name = f'./answered/{retriever_name}_top{args.ctx_topk}_{args.llm_name}_results.csv'
    to_save_df.to_csv(result_name, index=False, encoding='utf-8')
    print(f"Saved as {result_name}")

    ##########
    print('--- TimeQA ---')
    to_save_timeqa = [ex for ex in to_save if ex['source']=='timeqa']
    # separate samples into different types for comparison
    exact_param, exact_rag = [], []
    not_exact_param, not_exact_rag = [], []
    for example in to_save_timeqa:
        if example['time_relation'] == '':
            pass
        elif int(example['exact']) == 1:
            exact_rag.append(example['rag_acc'])
            if 'param_acc' in example:
                exact_param.append(example['param_acc'])
        else:
            not_exact_rag.append(example['rag_acc'])
            if 'param_acc' in example:
                not_exact_param.append(example['param_acc'])

    if args.param_pred:
        print('Parametric')
        print(f'    w/ key date acc : {round(np.mean(exact_param),4)}')
        print(f'    w/ perturb date acc : {round(np.mean(not_exact_param),4)}')

    print('RAG')
    print(f'    w/ key date acc : {round(np.mean(exact_rag),4)}')
    print(f'    w/ perturb date acc : {round(np.mean(not_exact_rag),4)}')

    ##########
    print('\n--- SituatedQA ---')
    to_save_situatedqa = [ex for ex in to_save if ex['source']=='situatedqa']
    # separate samples into different types for comparison
    exact_param, exact_rag = [], []
    not_exact_param, not_exact_rag = [], []
    for example in to_save_situatedqa:
        if example['time_relation'] == '':
            pass
        elif int(example['exact']) == 1:
            exact_rag.append(example['rag_acc'])
            if 'param_acc' in example:
                exact_param.append(example['param_acc'])
        else:
            not_exact_rag.append(example['rag_acc'])
            if 'param_acc' in example:
                not_exact_param.append(example['param_acc'])

    if args.param_pred:
        print('Parametric')
        print(f'    w/ key date acc : {round(np.mean(exact_param),4)}')
        print(f'    w/ perturb date acc : {round(np.mean(not_exact_param),4)}')

    print('RAG')
    print(f'    w/ key date acc : {round(np.mean(exact_rag),4)}')
    print(f'    w/ perturb date acc : {round(np.mean(not_exact_rag),4)}')

if __name__ == "__main__":
    main()