from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams
 
from temp_eval import normalize

def reader_pipeline(reader, llm, prompts):
    if reader in ['timo','timellama']:
        outputs = llm(prompts, do_sample=True, max_new_tokens=100, num_return_sequences=1, temperature=0.7, top_p=0.95)
        outputs = [r[0]['generated_text'] for r in outputs]
        responses = [outputs[i].replace(prompts[i],'') for i in range(len(prompts))]
    else:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
        outputs = llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
    responses = [res.split('<Question>:')[0] for res in responses]
    responses = [res.split('<doc>')[0] for res in responses]
    responses = [res.split('<Answer>:\n')[-1] for res in responses]
    responses = [res.split('\n')[0] for res in responses]
    return responses

def TIMO():
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Warrieryes/timo-13b-hf", device=0)
    return pipe

def TimeLLAMA():
    from transformers import pipeline
    # pipe = pipeline("text-generation", model="chrisyuan45/TimeLlama-13b", model_kwargs={'load_in_8bit':True})
    pipe = pipeline("text-generation", model="chrisyuan45/TimeLlama-7b", device=0)
    return pipe

# from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
# # Model names: "chrisyuan45/TimeLlama-7b-chat", "chrisyuan45/TimeLlama-13b-chat"
# model = LlamaForCausalLM.from_pretrained(
#         model_name,
#         return_dict=True,
#         load_in_8bit=quantization,
#         device_map="auto",
#         low_cpu_mem_usage=True)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)


def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_outputs.json")
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=5)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument('--param-cot', type=bool, default=True)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    # parser.add_argument('--ctx-key-s2', type=str, default='snt_hybrid_rank')
    parser.add_argument('--ctx-key-s2', type=str, default='reranker_ctxs')
    parser.add_argument('--reader', type=str, default='timellama', choices=['rag', 'metriever', 'timo', 'timellama', 'extract_code'], help="Choose a reader option")

    args = parser.parse_args()
    args.l = llm_names(args.llm)
    args.llm_name = deepcopy(args.llm)
    # if args.reader=='metriever':
    #     assert args.ctx_key_s2 == 'snt_hybrid_rank'

    if args.stage1_model=='bm25':
        args.ctx_key_s1 = 'bm25_ctxs'
    elif args.stage1_model=='hybrid':
        args.ctx_key_s1 = 'hybrid_ctxs'
    else:
        args.ctx_key_s1 = 'ctxs'

    # load llm
    if args.reader=='timo':
        args.llm = TIMO()
        args.llm_name = 'timo'
    elif args.reader=='timellama':
        args.llm = TimeLLAMA()
        args.llm_name = 'timellama'
    else:
        flg = '70b' in args.llm_name
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ", max_model_len=4096)
        else:
            args.llm = LLM(args.l, tensor_parallel_size=1, dtype='half', max_model_len=4096)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]
    
    ########  QA  ######## 
    if args.param_pred:
        if args.param_cot:
            prompts = [zc_cot_prompt(ex['question']) for ex in examples]
        else:
            prompts = [zc_prompt(ex['question']) for ex in examples]
        param_preds = reader_pipeline(args.reader, args.llm, prompts)
        print('zero context prediction finished.')

    tmp_key = args.ctx_key_s2 if args.ctx_key_s2 else args.ctx_key_s1
    # if args.metriever_reader:
    #     for ex in examples[1:2]:
    #         normalized_question = ex['normalized_question']
    #         QFS_prompts = [get_QFS_prompt(normalized_question, ctx['title'], ctx['text']) for ctx in ex[tmp_key][:args.ctx_topk]]
    #         summary_responses = call_pipeline(args, QFS_prompts)
    #         import ipdb; ipdb.set_trace()
    # else:
    if args.reader == 'metriever':
        pass
    elif args.reader == 'extract_code':
        prompts, texts = [], []
        for ex in examples:
            txts = [ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]]
            texts.append(txts)
            pmts = [extract_information_prompt(ex['question'], text) for text in txts]
            prompts += pmts
        ec_preds = reader_pipeline(args.reader, args.llm, prompts)
    else:
        prompts, texts = [], []
        for ex in examples:
            text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
            texts.append(text)
            prompt = c_prompt(ex['question'], text)
            prompts.append(prompt)
        rag_preds = reader_pipeline(args.reader, args.llm, prompts)
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

        if args.reader == 'extract_code':
            ec_pred = ec_preds[k*args.ctx_topk:(k+1)*args.ctx_topk]
            import ipdb; ipdb.set_trace()
        elif args.reader == 'metriever':
            pass
        else:
            rag_pred = rag_preds[k]
            ex['rag_pred'] = rag_pred
            ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
            ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
 
        for ctx_key in [args.ctx_key_s1, args.ctx_key_s2]:
            if 'QFS_summary' not in ex[ctx_key][0]:
                for ctx in ex[ctx_key]:
                    ctx['QFS_summary']=''

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
    result_name = f'./answered/{retriever_name}_top{args.ctx_topk}_{args.llm_name}_results.csv'
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