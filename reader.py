import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
    # parser.add_argument('--retriever-output', type=str, default="timeqa_contriever_bgegemma_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=3)
    parser.add_argument('--param-pred', type=bool, default=True)
    parser.add_argument('--param-cot', type=bool, default=False)
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--save-note', type=str, default='dp')
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--reader', type=str, default='timo', choices=['llama', 'timo', 'timellama','llama_70b','llama_8b'])
    parser.add_argument('--paradigm', type=str, default='concat', choices=['fusion', 'concat'])

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

    # examples = examples[100:200]
    # x = "When was the last time the Dodgers played the Yankees in the World Series between 1979 and 1999?"
    # examples = [ex for ex in examples if x in ex['question']]
    
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
        reader_prompts = []

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

                reader_prompt = reader(normalized_question, ctx['title'], ctx['text'])
                reader_prompts.append(reader_prompt)
        
        checker_responses = call_pipeline(args, checker_prompts, 500)
        checker_results = ['yes' in res.lower() for res in checker_responses]
        print('started reader')
        reader_responses = call_pipeline(args, reader_prompts, 500, return_list=True)

        # ensure the year appears in the responses also appears in the reader context
        tmp=[]
        for r_p, r_r in zip(reader_prompts, reader_responses):
            r_p = r_p.split('Now your context paragraph and question are')[-1]
            r_r = [snt for snt in r_r if year_identifier(snt)==None or all([str(y) in r_p for y in year_identifier(snt)])]
            tmp.append(r_r)
        reader_responses = tmp

        entail_check_map = {}
        contexts, anss = [], []
        entailer_prompts = []
        for reader_prompt, reader_response, checker_res in zip(reader_prompts, reader_responses, checker_results):
            if checker_res:
                reader_prompt = reader_prompt.split('Now your context paragraph and question are\n')[-1]
                context = reader_prompt.split('\n</Context>')[0].split('<Context>\n')[-1]
                for ans in reader_response:
                    contexts.append(context)
                    anss.append(ans)
                    entailer_prompt = entailer(context, ans)
                    entailer_prompts.append(entailer_prompt)
        entailer_responses = call_pipeline(args, entailer_prompts, 200)
        entailer_results = ['yes' in res.lower() for res in entailer_responses]
        for context, ans, entail in zip(contexts, anss, entailer_results):
            entail_check_map[context.strip()+'<>'+ans.strip()] = entail

        # import ipdb; ipdb.set_trace()

        for x, y, z in zip(reader_prompts, reader_responses, checker_responses):
            print('\n==\n')
            print(x.split('Now your context paragraph and question are')[-1])
            print(y)
            print(z)

        sub_examples = []
        timer_prompts = []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']
            reader_ans = []
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                checker_result = checker_results.pop(0)
                reader_response = reader_responses.pop(0)
                if checker_result and len(reader_response)>0:
                    print('xxx ', reader_response)
                    tmp = []
                    for ans in reader_response:
                        context = f"{ctx['title']} | {ctx['text']}"
                        key = context.strip()+'<>'+ans.strip()
                        if key in entail_check_map and entail_check_map[key]:
                            tmp.append(ans)
                    if tmp != reader_response:
                        print('yyy ', tmp)
                    reader_ans += tmp
            reader_ans = list(set(reader_ans))
            for reader_a in reader_ans:
                sub_examples.append([question, normalized_question, reader_a])
                timer_prompts.append(timer(normalized_question, reader_a))
        if len(sub_examples)>0:
            timer_responses = call_pipeline(args, timer_prompts, 100)
        else:
            timer_responses = []


        question_result_map = {}
        question_no_date_result_map = {}
        for sub_ex, timer_r in zip(sub_examples, timer_responses):
            try:
                answer_dict = eval(timer_r)
                assert isinstance(answer_dict, dict)
                if sub_ex[0] not in question_no_date_result_map:
                    question_no_date_result_map[sub_ex[0]] = []
                ans = list(answer_dict.keys())[0].strip()
                if len(ans)>0:
                    question_no_date_result_map[sub_ex[0]].append(ans)
                assert all([item in list(answer_dict.values())[0] for item in ["start_year", "start_month", "end_year", "end_month"]])
                # if there is no time info, skip
                assert sum(list(answer_dict.values())[0].values())>0
            except Exception as e:
                sub_years = year_identifier(sub_ex[2])
                print('ERROR: ', timer_r)
                if sub_years:
                    print(sub_years[0])
                    answer_dict = {str(sub_years[0]):{'start_year': sub_years[0], 'start_month': 0, 'end_year': sub_years[0], 'end_month': 0}}
                else:
                    continue
            sub_ex.append(answer_dict)
            if sub_ex[0] not in question_result_map:
                question_result_map[sub_ex[0]] = []
            question_result_map[sub_ex[0]].append(sub_ex)

        combiner_prompts=[]
        new_questions=[]
        for k, ex in enumerate(examples):
            question = ex['question']
            normalized_question = ex['normalized_question']
            if question not in question_result_map:
                continue
            result = question_result_map[question]

            print('\n------\n', k,' ',question,'\n------\n') 
        
            answer_dicts = [r[3] for r in result]
            append_flgs = []
            if ex['time_relation_type']=='before':
                q_year = ex['years'][0]
                q_month = ex['months'][0] if sum(ex['months'])>0 else None

                for answer_dict in answer_dicts:
                    answer_dict = list(answer_dict.values())[0]
                    start_year = answer_dict['start_year']
                    start_month = answer_dict['start_month']

                    append_flg = True
                    if start_year>0:
                        if start_year==q_year and q_month and start_month>0:
                            if start_month>q_month:
                                append_flg=False
                        else:
                            if ex['time_relation']=='before':
                                if start_year>=q_year:
                                    append_flg=False
                            else:
                                if start_year>q_year:
                                    append_flg=False

                    append_flgs.append(append_flg)
   
            elif ex['time_relation_type']=='after':
                q_year = ex['years'][0]
                q_month = ex['months'][0] if sum(ex['months'])>0 else None

                for answer_dict in answer_dicts:
                    answer_dict = list(answer_dict.values())[0]
                    end_year = answer_dict['end_year']
                    end_month = answer_dict['end_month']              
                    
                    append_flg = True
                    if end_year>0:
                        if end_year==q_year and q_month and end_month>0:
                            if ex['time_relation']=='since':
                                if end_month<q_month:
                                    append_flg=False
                            else:
                                if end_month<=q_month:
                                    append_flg=False
                        elif ex['time_relation']=='after':
                            if end_year<=q_year:
                                append_flg=False
                        else:
                            if end_year<q_year:
                                append_flg=False
                    append_flgs.append(append_flg)

            elif ex['time_relation_type']=='between':
                q_year_s = min(ex['years'])
                q_year_e = max(ex['years'])

                for answer_dict in answer_dicts:
                    answer_dict = list(answer_dict.values())[0]
                    start_year = answer_dict['start_year']
                    end_year = answer_dict['end_year']
                    
                    append_flg = True
                    if start_year>q_year_e:
                        append_flg = False
                    if end_year>0 and end_year<q_year_s:
                        append_flg = False

                    append_flgs.append(append_flg)
            else:
                append_flgs.append(True)


            filtered_result = [r for r, flg in zip(result, append_flgs) if flg]
            if len(filtered_result)==0:
                continue

            if len(filtered_result)>0:
                filtered_result = sorted(filtered_result, key=lambda x: (list(x[-1].values())[0]['start_year'], list(x[-1].values())[0]['start_month']), reverse=False)

            new_questions.append(question)
            if len(filtered_result)>10:
                filtered_result = filtered_result[:5]+filtered_result[-5:]
            contexts = '\n'.join([tp[2].replace(' first','').replace(' last','') for tp in filtered_result])


            combiner_prompt = combiner(question, contexts)
            combiner_prompts.append(combiner_prompt)
        
        if len(combiner_prompts)>0:
            combiner_responses = call_pipeline(args, combiner_prompts, 500)
            combiner_responses = [res.replace('\n','').strip() for res in combiner_responses]
        else:
            combiner_responses = []

        for x,y in zip(combiner_prompts, combiner_responses):
            print(x.split('Now your context paragraph and question are')[-1])
            print(y)

        question_answer_map = {q: a for q, a in zip(new_questions, combiner_responses) if len(a)>0}
        
        for k, ex in enumerate(examples):
            question = ex['question']
            find_flg = True
            if question in question_answer_map:
                rag_pred = question_answer_map[question]
                if any([item in rag_pred.lower() for item in ['unknown', 'none', 'not mention', 'not provide', "do not know"]]):
                    find_flg = False
            else:
                find_flg = False

            # pick from top candidate answer without dates
            if find_flg==False and question in question_no_date_result_map:
                tmp = question_no_date_result_map[question]
                if len(tmp)>0:
                    rag_pred = tmp[0]
                    find_flg = True

            if not find_flg:
                print('no context is useful.')
                prompt  = zc_cot_prompt(question)
                rag_pred = call_pipeline(args, [prompt], 400)[0]

            pred_years = year_identifier(rag_pred)
            if question.lower().startswith('when') and pred_years:
                rag_pred = str(pred_years[-1])

            # too many words for answer
            if len(rag_pred.split())>20:
                print('prametric knowledge does not know.')
                text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]])
                prompt = c_prompt(ex['question'], text)
                tmp = call_pipeline(args, [prompt], 100)
                rag_pred = tmp[0] if len(tmp)>0 else ''

            if '&amp;' in rag_pred:
                rag_pred = rag_pred.replace('&amp;','&')

            ex['rag_pred'] = rag_pred
            ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
            ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
            print(rag_pred, ex['answers'], ex['rag_acc'])


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