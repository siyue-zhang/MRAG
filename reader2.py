import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import ray
ray.init(num_gpus=4) 

from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse 
from temp_eval import normalize


# def TIMO():
#     from transformers import pipeline
#     pipe = pipeline("text-generation", model="Warrieryes/timo-13b-hf", model_kwargs={'load_in_8bit':True}, device_map='auto')
#     return pipe

# def TimeLLAMA():
#     from transformers import pipeline
#     pipe = pipeline("text-generation", model="chrisyuan45/TimeLlama-13b-chat", device_map='auto')
#     return pipe


def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--retriever-output', type=str, default="timeqa_contriever_metriever_bgegemma_llama_8b_qfs5_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="timeqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=5)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument('--param-cot', type=bool, default=False)
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--save-note', type=str, default="concat_filter")
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--reader', type=str, default='llama_8b', choices=['llama', 'timo', 'timellama','llama_70b','llama_8b'])
    parser.add_argument('--paradigm', type=str, default='fusion', choices=['fusion', 'concat'])

    args = parser.parse_args()
    assert args.stage1_model in args.retriever_output
    if args.reader == 'llama':
        args.reader = "llama_8b"
    
    if 'metriever' in args.retriever_output:
        args.ctx_key_s2 = 'snt_hybrid_rank'
    else:
        args.ctx_key_s2 = 'reranker_ctxs'

    if args.stage1_model=='bm25':
        args.ctx_key_s1 = 'bm25_ctxs'
    elif args.stage1_model=='hybrid':
        args.ctx_key_s1 = 'hybrid_ctxs'
    else:
        args.ctx_key_s1 = 'ctxs'

    args.llm_name = args.reader
    args.l = llm_names(args.reader, instruct=True)
    flg = '70b' in args.llm_name
    if flg:
        args.llm = LLM(args.l, tensor_parallel_size=4, quantization="AWQ", max_model_len=20000)
    else:
        mx_len = 2048 if args.reader=='timo' else 20000
        args.llm = LLM(args.l, tensor_parallel_size=4, max_model_len=mx_len)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]
    
    examples = [ex for ex in examples if ex['time_relation'] != '']
    if len(examples)==0:
        print(f'find no example in top {args.max_examples}.')
    
    ########  QA  ######## 
    if args.param_pred:
        if args.param_cot:
            prompts = [zc_cot_prompt(ex['question']) for ex in examples]
        else:
            prompts = [zc_prompt(ex['question']) for ex in examples]
        param_preds = call_pipeline(args, prompts, 400)
        print('zero context prediction finished.')

    tmp_key = args.ctx_key_s2 if args.ctx_key_s2 else args.ctx_key_s1

    if args.paradigm=='concat':

        prompts, texts = [], []
        for ex in examples:
            text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
            texts.append(text)
            # prompt = c_cot_prompt(ex['question'], text)
            prompt = c_prompt(ex['question'], text)
            prompts.append(prompt)

        rag_preds = call_pipeline(args, prompts, 500)

        print(f'{tmp_key} top {args.ctx_topk} contexts prediction finished.')
        # import ipdb; ipdb.set_trace()

        for k, ex in enumerate(examples):
            question = ex['question']
            gold_evidences = ex['gold_evidences']
            rag_pred = rag_preds[k]
            ex['rag_pred'] = rag_pred
            if isinstance(rag_pred, list):
                if len(rag_pred)>0:
                    rag_pred = str(rag_pred[0])
                else:
                    rag_pred = ''
            assert isinstance(rag_pred, str), rag_pred
            ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
            ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
    
    else: 
        # checker & reader
        print('\nstarted checker.\n')
        checker_prompts = []

        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']

            if question.startswith('How many'):
                if 'How many times' in question:
                    normalized_question = normalized_question.replace('How many times', 'When')
                else:
                    normalized_question = normalized_question.replace('How many', 'What')
                ex['normalized_question'] = normalized_question

            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                context = f"{ctx['title']} | {ctx['text']}"
                checker_prompt = checker(normalized_question, context)
                checker_prompts.append(checker_prompt)

        checker_responses = call_pipeline(args, checker_prompts, 500)
        checker_results = ['yes' in res.lower() for res in checker_responses]
        print('started reader')

        prompts, texts = [], []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']
            rel = []
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                checker_result = checker_results.pop(0)
                if checker_result:
                    rel.append(ctx)

            text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in rel])
            texts.append(text)
            prompt = c_prompt(ex['question'], text)
            prompts.append(prompt)

        rag_preds = call_pipeline(args, prompts, 500)

        for k, ex in enumerate(examples):
            question = ex['question']
            gold_evidences = ex['gold_evidences']
            rag_pred = rag_preds[k]
            ex['rag_pred'] = rag_pred
            if isinstance(rag_pred, list):
                if len(rag_pred)>0:
                    rag_pred = str(rag_pred[0])
                else:
                    rag_pred = ''
            assert isinstance(rag_pred, str), rag_pred
            ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
            ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
    

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

        for ctx_key in [args.ctx_key_s1, args.ctx_key_s2]:
            if 'QFS_summary' not in ex[ctx_key][0]:
                for ctx in ex[ctx_key]:
                    ctx['QFS_summary']=''

        s1_ctx_text = '\n\n'.join([f" {t+1} | {ctx['hasanswer']} | {ctx['title']} | {ctx['text']}" for t, ctx in enumerate(ex[args.ctx_key_s1][:20])])
        s2_ctx_text = '\n\n'.join([f" {t+1} | {ctx['hasanswer']} | {ctx['title']} | {ctx['text']}\nQFS: {ctx['QFS_summary']}" for  t, ctx in enumerate(ex[args.ctx_key_s2][:20])]) if args.ctx_key_s2 else ''
        
        result = {
            'id': ex['id'],
            'source': ex['source'],
            'question': question,
            'exact': ex['exact'],
            'answers': ex['answers'],
            'time_relation': ex['time_relation'],
            'retriever_ans_hit': ans_index,
            'retriever_gold_hit': gold_index,
            'retriever_ctxs': s1_ctx_text,
            'reranker_ans_hit': ans_index_reranker,
            'reranker_gold_hit': gold_index_reranker,
            'reranker_ctxs': s2_ctx_text,
            'rag_pred': ex['rag_pred'],
            'rag_acc': ex['rag_acc'],
            'rag_f1': ex['rag_f1'],
            'gold_evidence_1': gold_evidences[0] if len(gold_evidences)>0 else '',
            'gold_evidence_2': gold_evidences[1] if len(gold_evidences)>1 else '',
            'top_snts': ex['top_snts'] if 'top_snts' in ex else '',
            }

        if args.param_pred:
            param_pred = param_preds[k]
            ex['param_pred'] = param_pred
            ex['param_acc'] = int(normalize(param_pred) in [normalize(ans) for ans in ex['answers']])
            ex['param_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(param_pred))
            result.update({
                'param_pred': ex['param_pred'],
                'param_acc': ex['param_acc'],
                'param_f1': ex['param_f1']
            })

        to_save.append(result)



    to_save_df = pd.DataFrame(to_save)
    retriever_name = args.retriever_output.split('/')[-1].split('_outputs')[0]
    result_name = f'./answered/{retriever_name}_top{args.ctx_topk}_{args.llm_name}_{args.paradigm}_results.csv'
    if isinstance(args.save_note, str) and len(args.save_note)>0:            
        result_name = result_name.replace('_results', f'_{args.save_note}_results')
    if not args.not_save:
        to_save_df.to_csv(result_name, index=False, encoding='utf-8')
        print(f"Saved as {result_name}")



    ##########
    print('--- TimeQA ---')
    eval_reader(to_save, args.param_pred, subset='timeqa', metric='acc')
    eval_reader(to_save, args.param_pred, subset='timeqa', metric='f1')

    ##########
    print('\n--- SituatedQA ---')
    eval_reader(to_save, args.param_pred, subset='situatedqa', metric='acc')
    eval_reader(to_save, args.param_pred, subset='situatedqa', metric='f1')




if __name__ == "__main__":
    main()