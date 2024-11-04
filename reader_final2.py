
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse
from copy import deepcopy
# from vllm import LLM, SamplingParams
 
from temp_eval import normalize

def checker(question, context):
    prompt = f"""Consider the question and context paragraph below:

<Context>
{context}
</Context>
<Question>
{question}
</Question>

Does the context provide an answer to the question?

First, read the context and respond with either "Yes" or "No". Then, briefly explain your reasoning.

<Response>
"""
    return prompt


def reader(question, context, guidance):
    prompt = f"""You will be given a context paragraph and a question. First read the context paragraph carefully. Use this context information to answer the question accurately. Focus specifically on the context sentences related to the guidance if provided.
Requirements are follows:
- Ensure each answer is a text span from the context paragraph.
- For question starts with "when", ensure the answer is a date not a time, such as 3 December 2015, December 2015, or December 3, 2015.
- For question starts with "who", ensure the answer is a complete name.

There are some examples for you to refer to:
<Context>
History of the Dallas Cowboys | The first NFL team to win three Super Bowls in four years, with Super Bowl wins in the 1992, 1993, and 1995 seasons. Only one other team, the New England Patriots, have won three Super Bowls in a four-year time span, doing so in the 2001, 2003, and 2004 seasons. The first team to hold the opposing team to no touchdowns in a Super Bowl. Dallas beat the Miami Dolphins 24–3 in Super Bowl VI. The only other teams to do this are the New England Patriots, who did so in their 13–3 win against the Los Angeles Rams in Super Bowl LIII, and the Tampa Bay Buccaneers in Super Bowl LV, beating
</Context>
<Question>
For which NFL season did the Dallas Cowboys win the Super Bowl
</Question>
<Guidance>
- The first NFL team to win three Super Bowls in four years, with Super Bowl wins in the 1992–93, 1993, and 1995-1996 seasons. Only one other team, the New England Patriots, have won three Super Bowls in a four-year time span, doing so in the 2001, 2003, and 2004 seasons.
</Guidance>
<Answer>
- 1992–93
- 1993
- 1995-1996
</Answer>

<Context>
Oliver Bulleid | A brief period working for the Board of Trade followed from 1910, arranging exhibitions in Brussels, Paris and Turin. He was able to travel widely in Europe, later including a trip with Nigel Gresley, William Stanier and Frederick Hawksworth, to Belgium, in 1934, to see a metre-gauge bogie locomotive. In December 1912, he rejoined the GNR as Personal Assistant to Nigel Gresley, the new CME. Gresley was only six years Bulleid's senior.
</Context>
<Question>
Oliver Bulleid was an employee for whom
</Question>
<Guidance>
None
</Guidance>
<Answer>
- Board of Trade
- GNR
</Answer>

<Context>
List of presidents of India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India
</Question>
<Guidance>
- In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002.
- In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Guidance>
<Answer>
- Neelam Sanjiva Reddy
- K. R. Narayanan
- Droupadi Murmu
</Answer>

<Context>
Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Guidance>
- The Lost World: Jurassic Park is a 1997 American science fiction action film.
- It was a blockbuster hit and went on to gross over $914 million worldwide
</Guidance>
<Answer>
- $618.6 million
- $914 million
</Answer>

<Context>
1980 Summer Olympics | Hence, the selection process for the 1984 Summer Olympics consisted of a single finalized bid from Los Angeles, which the International Olympic Committee (IOC) accepted in 1976. Los Angeles was awarded the selection officially on May 18, 1978 for the 1984 Summer Olympics.
</Context>
<Question>
When has United States hosted Summer Olympics
</Question>
<Guidance>
- Los Angeles was awarded the selection officially on May 18, 1978 for the 1984 Summer Olympics.
</Guidance>
<Answer>
- 1984
</Answer>

Now your context paragraph, question, and guidance are as follows.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Guidance>
{guidance}
</Guidance>
<Answer>
"""
    return prompt


def timer(question, context, answer):
    prompt = f"""You will be given one context paragraph, one question, and one answer. First read the context carefully. Your task is to find the corresponding date for this answer. Your response should be in a python dict object.
- The date should be parsed into a python dict format with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date such as an event, write the same start and end time.
- If the answer applies "from" a specific date such as a status and a position, write this date as the start time and write "0" for the end time.
- If the answer applies "until" a specific date such as a status and a position, write this date as the end time and write "0" for the start time.

There are some examples for you to refer to:
<Context>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Context>
<Question>
Who served as President of India
</Question>
<Answer>
K. R. Narayanan
</Answer>
<Response>
{{"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}}}}
</Response>

<Context>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Context>
<Question>
Who served as President of India
</Question>
<Answer>
Droupadi Murmu
</Answer>
<Response>
{{"Droupadi Murmu": {{"start_year": 0, "start_month": 0, "end_year": 2024, "end_month": 0}}}}
</Response>

<Context>
The 1997 movie "The Lost World: Jurassic Park" grossed a total of $618.6 million at the worldwide box office
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
$618.6 million
</Answer>
<Response>
{{"$618.6 million": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}}}
</Response>

<Context>
The Houston Rockets won the two NBA championships (1994-97): 1994-1995, 1997.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
1997
</Answer>
<Response>
{{"1997": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}}}
</Response>

<Context>
The Houston Rockets won the two NBA championships in 1994-1995 and on July 14, 1997.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
July 14, 1997
</Answer>
<Response>
{{"July 14, 1997": {{"start_year": 1997, "start_month": 7, "end_year": 1997, "end_month": 7}}}}
</Response>

<Context>
The Houston Rockets won the NBA championship in 1994-1995, May 1995, and 1997.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
1994-1995
</Answer>
<Response>
{{"1994-1995": {{"start_year": 1994, "start_month": 0, "end_year": 1995, "end_month": 0}}}}
</Response>

<Context>
Grigsby volunteered for World War II in 1942 and served in the Army. Starting in 1946 Grigsby served as the Founder and Chair of the Art Department at Carver High School for eight years.
</Context>
<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Answer>
Army
</Answer>
<Response>
{{"Army": {{"start_year": 1942, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Response>

Now your question, context paragraph and answer are as follows.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Answer>
{answer}
</Answer>
<Response>
"""
    return prompt





def zc_cot_prompt(question):

    prompt=f"""As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your thought should start after <Thought> and end with </Thought>. Your answer should start after <Answer> and end with </Answer>.
There are some examples for you to refer to:

<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>
England has reached the semi-finals of FIFA World Cup in 1966, 1990, 2018. The latest year before 2019 is 2018. So the answer is 2018.
</Thought>
<Answer>
2018
</Answer>

<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Thought>
The last Super Bowl as of 2021 is Super Bowl LV, which took place in February 2021. In Super Bowl LV, the national anthem was performed by Eric Church and Jazmine Sullivan. So the answer is Eric Church and Jazmine Sullivan.
</Thought>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Question>
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Thought>
The last Rugby World Cup is held in 1987, 1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019. The last Rugby World Cup held between 2007 and 2016 is in 2015. The IRB 2015 Rugby World Cup was hosted by England. So the answer is England.
</Thought>
<Answer>
England
</Answer>

Now your Question is
<Question>
{question}
</Question>
<Thought>
"""
    return prompt




def c_prompt(query, texts):

    prompt=f"""Answer the given question, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
If the given knowledge does not contain the answer, answer the question with your own knowledge.

There are some examples for you to refer to:
<Context>
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>
2018
</Answer>

<Context>
Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</Context>
<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Context>
Rugby World Cup | Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

Rugby union | Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
</Context>
<Question>
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Answer>
England
</Answer>

<Context>
List of presidents of India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who served as President of India as of 10 Jan 2000
</Question>
<Answer>
K. R. Narayanan
</Answer>

Now your question and context knowledge are as follows.
<Context>
{texts}
</Context>
<Question>
{query}
</Question>
<Answer>:
"""
    return prompt





def TIMO():
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Warrieryes/timo-13b-hf", model_kwargs={'load_in_8bit':True}, device_map='auto')
    return pipe

def TimeLLAMA():
    from transformers import pipeline
    pipe = pipeline("text-generation", model="chrisyuan45/TimeLlama-13b", model_kwargs={'load_in_8bit':True}, device_map='auto')
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_incom_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=5)
    parser.add_argument('--param-pred', type=bool, default=True)
    parser.add_argument('--param-cot', type=bool, default=True)
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--no-guidance', type=bool, default=True)
    parser.add_argument('--save-note', type=str, default=None)

    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    # parser.add_argument('--ctx-key-s2', type=str, default='snt_hybrid_rank')
    # parser.add_argument('--ctx-key-s2', type=str, default='reranker_ctxs')
    parser.add_argument('--reader', type=str, default='llama', choices=['llama', 'timo', 'timellama'])
    parser.add_argument('--paradigm', type=str, default='concat', choices=['fusion', 'concat'])


    args = parser.parse_args()
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

    # load llm
    if args.reader=='timo':
        args.llm = TIMO()
        args.llm_name = 'timo'
    elif args.reader=='timellama':
        args.llm = TimeLLAMA()
        args.llm_name = 'timellama'
    else:
        args.llm_name = args.reader
        args.l = llm_names(args.reader, instruct=True)
        flg = '70b' in args.llm_name
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ", max_model_len=4096)
        else:
            args.llm = LLM(args.l, tensor_parallel_size=1, dtype='half', max_model_len=10000)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    if args.max_examples:
        # examples = examples[:min(len(examples),args.max_examples)]
        examples = examples[-args.max_examples:]
    
    # x = "Who won the latest America's Next Top Model as of 2021?"
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
        param_preds = call_pipeline(args, prompts, 200)
        print('zero context prediction finished.')

    tmp_key = args.ctx_key_s2 if args.ctx_key_s2 else args.ctx_key_s1

    if args.paradigm=='concat':

        prompts, texts = [], []
        for ex in examples:
            text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
            texts.append(text)
            prompt = c_prompt(ex['question'], text)
            prompts.append(prompt)

        rag_preds = call_pipeline(args, prompts)
        print(f'{tmp_key} top {args.ctx_topk} contexts prediction finished.')

        for k, ex in enumerate(examples):
            question = ex['question']
            gold_evidences = ex['gold_evidences']
            rag_pred = rag_preds[k]
            ex['rag_pred'] = rag_pred
            ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
            ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
    
    else: 
        # fusion reader
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

            ctx_id_to_snt = {}
            ctx_id_to_ctx = {}

            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                ctx_id = ctx['id']
                ctx_id_to_snt[ctx_id] = []
                ctx_id_to_ctx[ctx_id] = ctx
                
                context = f"{ctx['title']} | {ctx['text']}"
                checker_prompt = checker(normalized_question, context)
                checker_prompts.append(checker_prompt)

            for ctx_id, snt, _ in ex['top_snt_id']:
                if ctx_id not in ctx_id_to_snt:
                    break
                if ctx_id in ctx_id_to_snt:
                    title = ctx_id_to_ctx[ctx_id]['title']
                    text = ctx_id_to_ctx[ctx_id]['text']
                    if snt[:len(title)] == title:
                        snt = snt[len(title):].strip()
                    if snt in text and len(snt)<0.3*len(text):
                        ctx_id_to_snt[ctx_id].append(snt)
            
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                ctx_id = ctx['id']
                guidance_list = ctx_id_to_snt[ctx_id]
                if len(guidance_list)==0 or args.no_guidance:
                    guidance_text = 'None'
                else:
                    guidance_text = '\n'.join([f"- {x}" for x in guidance_list])
                context = f"{ctx['title']} {ctx['text']}"
                reader_prompt = reader(normalized_question, context, guidance_text)
                reader_prompts.append(reader_prompt)
            
        checker_responses = call_pipeline(args, checker_prompts, 50)
        checker_results = ['yes' in res.lower() for res in checker_responses]
        print('started reader')
        reader_responses = call_pipeline(args, reader_prompts, 100)

        for x, y, z in zip(reader_prompts, reader_responses, checker_responses):
            print('\n==\n')
            print(x.split('Now your context paragraph, question, and guidance are as follows.')[-1])
            print(y)
            print(z)

        # import ipdb; ipdb.set_trace()

        timer_prompts, new_questions, new_answers = [], [], []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']

            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                reader_res = reader_responses.pop(0)
                if checker_results.pop(0):
                    reader_res = [r.replace('\n','').strip() for r in reader_res]
                    reader_res = [r for r in reader_res if 'none' not in r.lower() and len(r)>0]
                    if len(reader_res)>0:
                        for res in reader_res:
                            context = f"{ctx['title']} {ctx['text']}"
                            timer_prompt = timer(normalized_question, context, res)
                            timer_prompts.append(timer_prompt)
                            new_answers.append(res)
                            new_questions.append(question)

        timer_responses = call_pipeline(args, timer_prompts, 100)

        for x,y in zip(timer_prompts, timer_responses):
            print('\n==\n')
            print(x.split('Now your question, context paragraph and answer are as follows.')[-1])
            print(y)
        # import ipdb; ipdb.set_trace()

        for j in range(len(timer_responses)):
            if new_questions[j].lower().startswith('when'):
                try:
                    eval(timer_responses[j])
                except Exception as e:
                    ys = year_identifier(new_answers[j])
                    if ys:
                        if len(ys)>1:
                            print('>1')
                            print('{"'+new_answers[j]+'": {"start_year": '+str(min(ys))+', "start_month": 0, "end_year": '+str(max(ys))+', "end_month": 0}'+'}')
                            timer_responses[j] = '{"'+new_answers[j]+'": {"start_year": '+str(min(ys))+', "start_month": 0, "end_year": '+str(max(ys))+', "end_month": 0}'+'}'
                        elif len(ys)==1:
                            print('=1')
                            print('{"'+new_answers[j]+'": {"start_year": '+str(ys[0])+', "start_month": 0, "end_year": '+str(ys[0])+', "end_month": 0}'+'}')
                            timer_responses[j] = '{"'+new_answers[j]+'": {"start_year": '+str(ys[0])+', "start_month": 0, "end_year": '+str(ys[0])+', "end_month": 0}'+'}'
                        # import ipdb; ipdb.set_trace()

        question_answer_map={}
        for q, res in zip(new_questions, timer_responses):
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


        for k, ex in enumerate(examples):
            question = ex['question']
            normalized_question = ex['normalized_question']
            gold_evidences = ex['gold_evidences']
            answer_dict = question_answer_map[question] if question in question_answer_map else {}

            print('\n------\n', k,' ',question,'\n------\n') 
            print('\nanswer dict')
            print(answer_dict)

            # import ipdb; ipdb.set_trace()

            tmp = []
            if ex['time_relation_type']=='before':
                q_year = ex['years'][0]
                q_month = ex['months'][0] if sum(ex['months'])>0 else None

                for ans in answer_dict:
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
                        elif ex['time_relation']=='after':
                            if end_year<=q_year:
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

            else:
                tmp = [{ans:answer_dict[ans]} for ans in answer_dict]

            # filter all zero
            ans_list, backup = [], []
            for ans_dict in tmp:
                ans = next(iter(ans_dict))
                flg = sum(list(ans_dict[ans].values()))==0
                # the answer of "when" question must have at least one "year"
                if normalized_question.lower().startswith('when') and year_identifier(list(ans_dict.keys())[0])==None:
                    pass
                elif ans !='0' and flg:
                    backup.append(ans_dict)
                elif not flg:
                    ans_list.append(ans_dict)

            print('\nafter filter')
            print(ans_list)
            print('\nbackup')
            print(backup)

            if len(ans_list)>0:
                if ex['implicit_condition'] == 'last':
                    ans_list = sorted(ans_list, key=lambda x: (list(x.values())[0]['start_year'], list(x.values())[0]['end_year'], list(x.values())[0]['start_month']), reverse=True)
                elif ex['implicit_condition'] == 'first':
                    ans_list = sorted(ans_list, key=lambda x: (list(x.values())[0]['start_year'], list(x.values())[0]['start_month']), reverse=False)
                else:
                    if ex['time_relation_type']=='between':
                        ans_list = sorted(ans_list, key=lambda x: (abs(list(x.values())[0]['start_year']-ex['years'][0]), abs(list(x.values())[0]['end_year']-ex['years'][1])), reverse=False)
                    else:
                        ans_list = sorted(ans_list, key=lambda x: abs(list(x.values())[0]['start_year']-ex['years'][0]), reverse=False)
                # in/on
                if ex['time_relation_type']=='other':
                    for ans_dict in ans_list:
                        ans = next(iter(ans_dict))
                        if ans_dict[ans]['start_year']>0:
                            if ex['years'][0] > ans_dict[ans]['start_year']:
                                if ex['years'][0] < ans_dict[ans]['end_year']:
                                    ans_list = [ans_dict]
                                    break

                print('\nafter sort')
                print(ans_list)

            if len(ans_list)==0:
                flg = True
                if len(backup)>0:
                    candidate = str(next(iter(backup[0])))
                    candidate = candidate.strip()
                    if len(candidate)>0 and candidate!='0':
                        flg=False
                        rag_pred=candidate
                if flg:
                    print('no context is useful.')
                    prompt  = zc_cot_prompt(question)
                    rag_pred = call_pipeline(args, [prompt])[0]
            else:
                rag_pred = next(iter(ans_list[0]))
                if question.lower().startswith('when'):
                    can1 = ans_list[0][rag_pred]['start_year']
                    can2 = ans_list[0][rag_pred]['end_year']
                    if ex['implicit_condition'] == 'last' and can2>0:
                        rag_pred = str(can2)
                    else:
                        rag_pred = str(can1)
                elif question.startswith('How many times'):
                    tmp = []
                    for ans_dict in ans_list:
                        ans = next(iter(ans_dict))
                        if ans_dict[ans]['start_year']==ans_dict[ans]['end_year']:
                            if ans_dict[ans]['start_year']>0:
                                ans_dict[ans]['start_month']=0
                                ans_dict[ans]['end_month']=0
                                ans_dict['ans'] = ans_dict[ans]
                                del ans_dict[ans]
                                if ans_dict not in tmp:
                                    tmp.append(ans_dict)
                    print('\nHOW MANY TIMES: ', question)
                    print(tmp)
                    rag_pred = str(len(tmp))
                elif question.startswith('How many'):
                    tmp = []
                    for ans_dict in ans_list:
                        ans = next(iter(ans_dict))
                        if ans_dict[ans]['start_year']>0 and ans_dict[ans]['end_year']>0:
                            ans_dict[ans]['start_month']=0
                            ans_dict[ans]['end_month']=0
                            ans_dict['ans'] = ans_dict[ans]
                            del ans_dict[ans]
                            if ans_dict not in tmp:
                                tmp.append(ans_dict)
                    print('\nHOW MANY: ', question)
                    print(tmp)
                    rag_pred = str(len(tmp))
                    
            rag_pred = rag_pred.replace('\n','').strip()

            if len(rag_pred)==0:
                text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]])
                prompt = c_prompt(ex['question'], text)
                rag_pred = call_pipeline(args, [prompt])[0]
                rag_pred = rag_pred.replace('\n','').strip()
            
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
    if args.save_note:            
        result_name = result_name.replace('_outputs', f'_{args.save_note}_outputs')
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