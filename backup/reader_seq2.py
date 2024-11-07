
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams
 
from temp_eval import normalize

def checker(question, context):
    prompt = f"""Does the context paragraph contain the answer to the question? Response Yes or No.
There are some examples for you to refer to:
<Context>
J. Eugene Grigsby | Grigsby volunteered for World War II in 1942 and served in the Army.
</Context>
<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Response>
Yes
</Response>

<Context>
NBA Finals | Lakers won the NBA championship in 2007.
</Context>
<Question>
The time when the Houston Rockets won the NBA championship
</Question>
<Response>
No
</Response>

Now your context paragraph and question are as follows.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Response>
"""
    return prompt

def reader(question, title, text):
    prompt = f"""You will be given a context paragraph and a question. Your task is to write independent sentences to answer the question based on the context paragraph. 
Requirements are follows:
- Each independent sentence should be standalone with specific subjects, objects, relations, and actions.
- Each independent sentence should include the date if it is mentioned or can be inferred in the context paragraph.

There are some examples for you to refer to:
<Context>:
Houston Rockets | The team have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
- The Houston Rockets won the NBA championship in 1994 ans 1995.
</Answer>

<Context>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India
</Question>
<Answer>
- Neelam Sanjiva Reddy served as the sixth President of India from 1977.
- K. R. Narayanan became the first Dalit to serve as the President of India from 1997 until 2002.
- Droupadi Murmu served as the 15th President of India from 2022.
</Answer>

<Context>
The Lost World: Jurassic Park | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America.
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
- The movie, The Lost World: Jurassic Park, grossed a total of $618.6 million at the worldwide box office in 1997.
</Answer>

Now your context paragraph and question are as follows.
<Context>
{title} | {text}
</Context>
<Question>
{question}
</Question>
<Answer>
"""
    return prompt

def formatter(question, sentence):
    prompt = f"""You will be given a question and several sentences. Your task is to find the answer and the corresponding date from the sentences.
- The result should be in the python dict format: the answer is the dict key and the corresponding date is the dict value.
- The answer should be a text span extracted from the sentences.
- The date should be parsed into a python dict object with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date, write the same start and end time.
- If the answer applies from a specific date, write this date as the start time and write "0" for the end time.
- If the answer applies until a specific date, write this date as the end time and write "0" for the start time.
- Write "0" if the date data is not available.

There are some examples for you to refer to:
<Sentence>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Sentence>
<Question>
Who served as President of India
</Question>
<Answer>
{{
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}},
"Droupadi Murmu": {{"start_year": 0, "start_month": 0, "end_year": 2024, "end_month": 0}}
}}
</Answer>

<Sentence>
The movie "The Lost World: Jurassic Park" grossed a total of $618.6 million at the worldwide box office in 1997.
</Sentence>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
{{
"$618.6 million": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}
}}
</Answer>

<Sentence>
The Houston Rockets won the NBA championship in 1994 and May 1995.
</Sentence>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
{{
"1994": {{"start_year": 1994, "start_month": 0, "end_year": 1994, "end_month": 0}},
"May 1995": {{"start_year": 1995, "start_month": 5, "end_year": 1995, "end_month": 5}}
}}
</Answer>

<Sentence>
Neelam Sanjiva Reddy served as the sixth President of India from Dec 1977, K. R. Narayanan - President of India (1997-98).
</Sentence>
<Question>
Who serve as President of India
</Question>
<Answer>
{{
"Neelam Sanjiva": {{"start_year": 1977, "start_month": 12, "end_year": 0, "end_month": 0}},
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 1998, "end_month": 0}}
}}
</Answer>

<Sentence>
Grigsby volunteered for World War II in 1942 and served in the Army. Starting in 1946 Grigsby served as the Founder and Chair of the Art Department at Carver High School for eight years.
</Sentence>
<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Answer>
{{
"Army": {{"start_year": 1942, "start_month": 0, "end_year": 0, "end_month": 0}},
"Carver High School": {{"start_year": 1946, "start_month": 0, "end_year": 1954, "end_month": 0}}
}}
</Answer>

Now your context sentence and question are as follows.
<Sentence>
{sentence}
</Sentence>
<Question>
{question}
</Question>
<Answer>
"""
    return prompt


def call_pipeline(args, prompts, max_tokens=100):
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max_tokens)
    outputs = args.llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    for stopper in ['</Keywords>', '</Summarization>', '</Answer>', '</Info>', '</Sentences>', '</Sentence>', '</Response>']:
        responses = [res.split(stopper)[0] if stopper in res else res for res in responses]
    for mid_stopper in ['</Thought>']:
        responses = [res.split(mid_stopper)[-1] if mid_stopper in res else res for res in responses]
    if '- ' in responses[0]:
        responses = [res.split('- ') for res in responses]
        tmp = []
        for res in responses:
            res = [r.replace('\n','').strip() for r in res]
            tmp.append([r for r in res if r !=''])
        responses = tmp
    return responses 

def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=5)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument('--param-cot', type=bool, default=True)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--ctx-key-s2', type=str, default='snt_hybrid_rank')
    # parser.add_argument('--ctx-key-s2', type=str, default='reranker_ctxs')
    parser.add_argument('--reader', type=str, default='seq', choices=['rag', 'timo', 'timellama', 'seq'], help="Choose a reader option")
    parser.add_argument('--temporal-filter', type=bool, default=False)

    args = parser.parse_args()
    args.l = llm_names(args.llm, instruct=True)
    args.llm_name = deepcopy(args.llm)
    if args.reader=='seq':
        args.llm_name += '_seq'

    if args.stage1_model=='bm25':
        args.ctx_key_s1 = 'bm25_ctxs'
    elif args.stage1_model=='hybrid':
        args.ctx_key_s1 = 'hybrid_ctxs'
    else:
        args.ctx_key_s1 = 'ctxs'


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
    

    ##


    ##

    
    # checker
    print('\nstarted checker.\n')
    checker_prompts = []
    for ex in examples:
        normalized_question = ex['normalized_question']
        for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
            prompt = checker(normalized_question, ctx['title']+' | '+ctx['text'])
            checker_prompts.append(prompt)
    checker_responses = call_pipeline(args, checker_prompts, 10)
    checker_results = [res[:3].lower()=='yes' for res in checker_responses]

    # reader
    print('\nstarted reader.\n')
    reader_prompts = []
    for ex in examples:
        normalized_question = ex['normalized_question']
        for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
            if 'when' not in normalized_question.lower():
                tmp = normalized_question + ' and when'
            else:
                tmp = normalized_question
            prompt = reader(tmp, ctx['title'], ctx['text'])
            reader_prompts.append(prompt)
    reader_responses = call_pipeline(args, reader_prompts, 500)

    # formatter
    print('\nstarted formatter.\n')
    formatter_prompts = []
    questions = []
    for k, ex in enumerate(examples):
        normalized_question = ex['normalized_question']
        question = ex['question']

        rel_events = []
        checker_r = checker_results[k*args.ctx_topk:(k+1)*args.ctx_topk]
        reader_r = reader_responses[k*args.ctx_topk:(k+1)*args.ctx_topk]
        for j in range(args.ctx_topk):
            if checker_r[j] and len(reader_r[j])>0 and reader_r[j] not in rel_events:
                rel_events += reader_r[j]

        while rel_events:
            batch = []
            for _ in range(1):
                if rel_events:
                    batch.append(rel_events.pop(0))
            prompt = formatter(normalized_question, ' '.join(batch))
            formatter_prompts.append(prompt)
            questions.append(question)
    formatter_responses = call_pipeline(args, formatter_prompts, 500)
    
    question_answer_map={}
    for q, res in zip(questions, formatter_responses):
        if q not in question_answer_map:
            question_answer_map[q] = {}
        try:
            answer_dict_b = eval(res)
            for ans in answer_dict_b:
                key_names = ['start_year', 'start_month', 'end_year', 'end_month']
                flg = [key in answer_dict_b[ans] for key in key_names]
                if all(flg) == True:
                    flg = True
                    for key in key_names:
                        if not isinstance(answer_dict_b[ans][key], int):
                            try:
                                answer_dict_b[ans][key] = int(answer_dict_b[ans][key])
                            except Exception as e:
                                flg=False
                                break
                    if flg:
                        if ans in question_answer_map[q]:
                            exist_date = question_answer_map[q][ans]
                            new_date = answer_dict_b[ans]
                            if new_date != exist_date:
                                if exist_date['start_year']==0 and new_date['start_year']>0:
                                    question_answer_map[q][ans]['start_year'] = new_date['start_year']
                                    question_answer_map[q][ans]['start_month'] = new_date['start_month']
                                if exist_date['end_year']==0 and new_date['end_year']>0:
                                    question_answer_map[q][ans]['end_year'] = new_date['end_year']
                                    question_answer_map[q][ans]['end_month'] = new_date['end_month']
                                # import ipdb; ipdb.set_trace()
                        else:
                            question_answer_map[q][ans] = answer_dict_b[ans]

        except Exception as e:
            pass
    
    # 

    to_save=[]
    for k, ex in enumerate(examples):
        if ex['time_relation'] == '':
            continue

        question = ex['question']
        normalized_question = ex['normalized_question']
        gold_evidences = ex['gold_evidences']
        answer_dict = question_answer_map[question] if question in question_answer_map else {}

        # if question != "For which NFL season did the Dallas Cowboys win their most recent Super Bowl as of August 2, 1995?":
        #     continue

        print('\n------\n', k,' ',question,'\n------\n') 
        
        print('\nanswer dict')
        print(answer_dict)

        # take out all 0
        backup = {ans:answer_dict[ans] for ans in answer_dict if sum(answer_dict[ans].values())==0}
        answer_dict = {ans:answer_dict[ans] for ans in answer_dict if sum(answer_dict[ans].values())>0}

        # rel_events = []
        # for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
        #     prompt = checker(normalized_question, ctx['title']+' | '+ctx['text'])
        #     responses = call_pipeline(args, [prompt], 10)
        #     response = responses[0]
        #     if response[:3].lower()=='yes':
        #         if 'when' not in normalized_question.lower():
        #             tmp = normalized_question + ' and when'
        #         else:
        #             tmp = normalized_question
        #         prompt = reader(tmp, ctx['title'], ctx['text'])
        #         responses = call_pipeline(args, [prompt], 500)
        #         print('\n',ctx['title'], ' | ', ctx['text'])
        #         print('-->')
        #         print(responses[0],'\n')
        #         rel_events += responses[0]

        # answer_dict = {}
        # while rel_events:
        #     batch = []
        #     for _ in range(4):
        #         if rel_events:
        #             batch.append(rel_events.pop(0))
        #     prompt = formatter(normalized_question, ' '.join(batch))
        #     responses = call_pipeline(args, [prompt], 500)
        #     response = responses[0]

        #     print(batch,'\n')
        #     print('~~>\n', response,'\n')
        #     try:
        #         answer_dict_b = eval(response)
        #         tmp = {}
        #         for ans in answer_dict_b:
        #             key_names = ['start_year', 'start_month', 'end_year', 'end_month']
        #             flg = [key in answer_dict_b[ans] for key in key_names]
        #             if all(flg) == True:
        #                 flg = True
        #                 for key in key_names:
        #                     if not isinstance(answer_dict_b[ans][key], int):
        #                         try:
        #                             answer_dict_b[ans][key] = int(answer_dict_b[ans][key])
        #                         except Exception as e:
        #                             flg=False
        #                             break
        #                 if flg:
        #                     tmp[ans] = answer_dict_b[ans]
        #         answer_dict.update(tmp)
        #     except Exception as e:
        #         pass

        tmp = []
        if ex['time_relation_type']=='before':
            q_year = ex['years'][0]
            q_month = ex['months'][0] if sum(ex['months'])>0 else None

            for ans in answer_dict:
                print('check: ', answer_dict[ans])
                start_year = answer_dict[ans]['start_year']
                start_month = answer_dict[ans]['start_month']

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

                if append_flg:
                    tmp.append({ans: answer_dict[ans]})

        elif ex['time_relation_type']=='after':
            q_year = ex['years'][0]
            q_month = ex['months'][0] if sum(ex['months'])>0 else None

            for ans in answer_dict:
                end_year = answer_dict[ans]['end_year']
                end_month = answer_dict[ans]['end_month']              
                
                append_flg = True
                if end_year>0:
                    if end_year==q_year and q_month and end_month>0:
                        if ex['time_relation']=='since':
                            if end_month<q_month:
                                append_flg=False
                        else:
                            if end_month<=q_month:
                                append_flg=False
                    else:
                        if end_year<q_year:
                            append_flg=False
                
                if append_flg:
                    tmp.append({ans: answer_dict[ans]})

        elif ex['time_relation_type']=='between':
            q_year_s = min(ex['years'])
            q_year_e = max(ex['years'])

            for ans in answer_dict:
                start_year = answer_dict[ans]['start_year']
                end_year = answer_dict[ans]['end_year']
                
                append_flg = True
                if start_year>q_year_e:
                    append_flg = False
                if end_year>0 and end_year<q_year_s:
                    append_flg = False
                
                if append_flg:
                    tmp.append({ans: answer_dict[ans]})

        ans_list = tmp
    
        print('\nafter filter')
        print(ans_list)

        if len(ans_list)>0:
            if ex['implicit_condition'] == 'last':
                ans_list = sorted(ans_list, key=lambda x: (list(x.values())[0]['start_year'], list(x.values())[0]['start_month']), reverse=True)
            elif ex['implicit_condition'] == 'first':
                ans_list = sorted(ans_list, key=lambda x: (list(x.values())[0]['start_year'], list(x.values())[0]['start_month']), reverse=False)
            else:
                # for rest, look for closest date
                ans_list = sorted(ans_list, key=lambda x: abs(list(x.values())[0]['start_year']-ex['years'][0]), reverse=False)

            print('\nafter sort')
            print(ans_list)
        

        if len(ans_list)==0:
            print('no context is useful.')
            prompt  = zc_prompt(question)
            rag_pred = call_pipeline(args, [prompt])[0]
            rag_pred = rag_pred.replace('\n','').strip()
            if rag_pred=='':
                print('parametric no answer.')
                if len(backup)>0:
                    rag_pred = list(backup.keys())[0]
                if rag_pred=='':
                    ctxs = ex['snt_hybrid_rank'][:args.ctx_topk]
                    texts = [ctx['title']+' | '+ctx['text'] for ctx in ctxs]
                    prompt  = c_prompt(question, '\n'.join(texts))
                    rag_pred = call_pipeline(args, [prompt])[0]
        else:
            rag_pred = next(iter(ans_list[0]))
            tmp = str(ans_list[0][rag_pred]['start_year'])
            if tmp in rag_pred:
                rag_pred = tmp
        print(rag_pred, ex['answers'])

        
        ex['rag_pred'] = rag_pred
        ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
        ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))

        


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

        # if args.param_pred:
        #     param_pred = param_preds[k]
        #     ex['param_pred'] = param_pred
        #     ex['param_acc'] = int(normalize(param_pred) in [normalize(ans) for ans in ex['answers']])
        #     ex['param_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(param_pred))
        #     result.update({
        #         'param_pred': ex['param_pred'],
        #         'param_acc': ex['param_acc'],
        #         'param_f1': ex['param_f1']
        #     })


        to_save.append(result)


    to_save_df = pd.DataFrame(to_save)
    retriever_name = args.retriever_output.split('/')[-1].split('_outputs')[0]
    result_name = f'./answered/{retriever_name}_top{args.ctx_topk}_{args.llm_name}_results.csv'
    to_save_df.to_csv(result_name, index=False, encoding='utf-8')
    print(f"Saved as {result_name}")


    eval_reader(to_save, False, subset='situatedqa', metric='acc')
    eval_reader(to_save, False, subset='situatedqa', metric='f1')








if __name__ == "__main__":
    main()