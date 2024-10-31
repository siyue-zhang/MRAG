
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
    prompt = f"""Consider the question and sentence below:

<Question>
{question}
<\Question>
<Sentence>
{context}
</Sentence>

Does the sentence provide an answer to the question?

First, respond with either "Yes" or "No." If you are not sure, response "No". Then, briefly explain your reasoning, focusing on how the sentence content relates to the question.

<Response>
"""
    return prompt


# def checker(question, context):
#     prompt = f"""Can you find at least one answer to the question from the context paragph? Response Yes or No.
# There are some examples for you to refer to:
# <Context>
# J. Eugene Grigsby | Grigsby volunteered for World War II in 1942 and served in the Army.
# </Context>
# <Question>
# J. Eugene Grigsby was an employee for whom
# </Question>
# <Response>
# Yes
# </Response>

# <Context>
# Dallas Cowboys | Cowboys routed the Chicago Bears 37–7 and Minnesota Vikings 23–6 before defeating the Denver Broncos 27–10 in Super Bowl XII in New Orleans. As a testament to Doomsday's dominance in the hard-hitting game, defensive linemen Randy White and Harvey Martin were named co-Super Bowl MVPs, the first and only time multiple players have received the award.
# </Context>
# <Question>
# For which NFL season did the Dallas Cowboys win the Super Bowl
# </Question>
# <Response>
# No
# </Response>

# <Context>
# NBA Finals | Lakers won the NBA championship in 2007.
# </Context>
# <Question>
# The time when the Houston Rockets won the NBA championship
# </Question>
# <Response>
# No
# </Response>

# <Context>
# History of the Dallas Cowboys | The first NFL team to win three Super Bowls in four years, with Super Bowl wins in the 1992, 1993, and 1995 seasons. Only one other team, the New England Patriots, have won three Super Bowls in a four-year time span, doing so in the 2001, 2003, and 2004 seasons. The first team to hold the opposing team to no touchdowns in a Super Bowl. Dallas beat the Miami Dolphins 24–3 in Super Bowl VI. The only other teams to do this are the New England Patriots, who did so in their 13–3 win against the Los Angeles Rams in Super Bowl LIII, and the Tampa Bay Buccaneers in Super Bowl LV, beating
# </Context>
# <Question>
# For which NFL season did the Dallas Cowboys win the Super Bowl
# </Question>
# <Response>
# Yes
# </Response>

# Now your context paragraph and question are as follows.
# <Context>
# {context}
# </Context>
# <Question>
# {question}
# </Question>
# <Response>
# """
#     return prompt

def reader(question, context):
    prompt = f"""You will be given a context paragraph and a question. Your task is to find the answer to the question in the context paragraph. 
Requirements are follows:
- Ensure each answer is a text span from the context paragraph.
- For "when" question, ensure the answer is a date such as year, month, and day.
- For "who" question, ensure the answer is a name such as a person, an organization, or a company.

There are some examples for you to refer to:
<Context>
History of the Dallas Cowboys | The first NFL team to win three Super Bowls in four years, with Super Bowl wins in the 1992, 1993, and 1995 seasons. Only one other team, the New England Patriots, have won three Super Bowls in a four-year time span, doing so in the 2001, 2003, and 2004 seasons. The first team to hold the opposing team to no touchdowns in a Super Bowl. Dallas beat the Miami Dolphins 24–3 in Super Bowl VI. The only other teams to do this are the New England Patriots, who did so in their 13–3 win against the Los Angeles Rams in Super Bowl LIII, and the Tampa Bay Buccaneers in Super Bowl LV, beating
</Context>
<Question>
For which NFL season did the Dallas Cowboys win the Super Bowl
</Question>
<Answer>
- 1992
- 1993
- 1995
</Answer>

<Context>
List of presidents of India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India
</Question>
<Answer>
- Neelam Sanjiva Reddy
- K. R. Narayanan
- Droupadi Murmu
</Answer>

<Context>
Oliver Bulleid | A brief period working for the Board of Trade followed from 1910, arranging exhibitions in Brussels, Paris and Turin. He was able to travel widely in Europe, later including a trip with Nigel Gresley, William Stanier and Frederick Hawksworth, to Belgium, in 1934, to see a metre-gauge bogie locomotive. In December 1912, he rejoined the GNR as Personal Assistant to Nigel Gresley, the new CME. Gresley was only six years Bulleid's senior.
</Context>
<Question>
Oliver Bulleid was an employee for whom
</Question>
<Answer>
- Board of Trade
- GNR
</Answer>

<Context>
Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
- $618.6 million
- $914 million
</Answer>

Now your context paragraph and question are as follows.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Answer>
"""
    return prompt

def timer(question, context, answer):
    prompt = f"""You will be given a question, a context paragraph and one answer. Your task is to find the corresponding date for this answer. Your response should be in a python dict object.
- The date should be parsed into a python dict format with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date such as an event, write the same start and end time.
- If the answer applies "from" a specific date such as a status and a position, write this date as the start time and write "0" for the end time.
- If the answer applies "until" a specific date such as a status and a position, write this date as the end time and write "0" for the start time.

There are some examples for you to refer to:
<Question>
Who served as President of India
</Question>
<Context>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Context>
<Answer>
K. R. Narayanan
</Answer>
<Response>
{{"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}}}}
</Response>

<Question>
Who served as President of India
</Question>
<Context>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Context>
<Answer>
Droupadi Murmu
</Answer>
<Response>
{{"Droupadi Murmu": {{"start_year": 0, "start_month": 0, "end_year": 2024, "end_month": 0}}}}
</Response>

<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Context>
The 1997 movie "The Lost World: Jurassic Park" grossed a total of $618.6 million at the worldwide box office
</Context>
<Answer>
$618.6 million
</Answer>
<Response>
{{"$618.6 million": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}}}
</Response>

<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Context>
The Houston Rockets won the NBA championship in 1994 and May 1995.
</Context>
<Answer>
May 1995
</Answer>
<Response>
{{"May 1995": {{"start_year": 1995, "start_month": 5, "end_year": 1995, "end_month": 5}}}}
</Response>

<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Context>
Grigsby volunteered for World War II in 1942 and served in the Army. Starting in 1946 Grigsby served as the Founder and Chair of the Art Department at Carver High School for eight years.
</Context>
<Answer>
Army
</Answer>
<Response>
{{"Army": {{"start_year": 1942, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Response>

Now your question, context paragraph and answer are as follows.
<Question>
{question}
</Question>
<Context>
{context}
</Context>
<Answer>
{answer}
</Answer>
<Response>
"""
    return prompt


def formatter(question, sentence):
    prompt = f"""You will be given a question and several sentences. Your task is to extract the answer and the corresponding date from the sentences.
- The result should be in the python dict format: the extracted answer is the dict key and the corresponding date is the dict value.
- Ensure the dict key is the answer to the question such as a name, date, organization, etc.
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
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 1998, "end_month": 0}},
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
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=max_tokens)
    outputs = args.llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    for stopper in ['</Keywords>', '</Summarization>', '</Answer>', '</Info>', '</Sentences>', '</Sentence>', '</Response>','</Date>']:
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
    parser.add_argument('--max-examples', type=int, default=100)
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

    
    # x = "Who does Bryce Dallas Howard play in The Grinch in 2000?"
    # examples = [ex for ex in examples if ex['question']==x]

    
    for ex in examples:
        ids = []
        snts = []
        normalized_question = ex['normalized_question']
        ctx_id_map={}
        for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
            ctx_id = ctx['id']
            ids.append(ctx_id)
            if ctx_id not in ctx_id_map:
                ctx_id_map[ctx_id] = ctx
        ex['ctx_id_map'] = ctx_id_map
        
        for ctx_id, snt, _ in ex['top_snt_id']:
            if ctx_id not in ids:
                break
            snt = snt[len(ctx_id_map[ctx_id]['title']):].strip()
            snts.append(snt)
        ex['snts'] = snts
        ex['ids'] = ids


    # checker
    print('\nstarted checker.\n')
    checker_prompts = []
    reader_prompts = []
    questions = []
    norm_questions = []
    contexts = []
    for ex in examples:
        question = ex['question']
        normalized_question = ex['normalized_question']
        ctx_id_map = ex['ctx_id_map']
        for snt, id in zip(ex['snts'], ex['ids']):
            questions.append(question)
            norm_questions.append(normalized_question)
            # context = f"{ctx_id_map[id]['title']} | {snt}"
            context = snt
            contexts.append(context)
            reader_prompts.append(reader(normalized_question, context))
            checker_prompts.append(checker(normalized_question, snt))
            
    checker_responses = call_pipeline(args, checker_prompts, 100)
    checker_results = ['yes' in res.lower() for res in checker_responses]
    reader_responses = call_pipeline(args, reader_prompts, 400)

    for j in range(len(reader_prompts)):
        reader_response = reader_responses[j]
        context = reader_prompts[j].split('<Context>')[-1]
        reader_responses[j] = [r for r in reader_response if r in context]

    # for x, y, z in zip(reader_prompts, reader_responses, checker_responses):
    #     print('\n==\n')
    #     print(x.split('Now your context paragraph and question are as follows.')[-1])
    #     print(y)
    #     print(z)

    # import ipdb; ipdb.set_trace()

    questions = [q for q, cond in zip(questions, checker_results) if cond]
    norm_questions = [q for q, cond in zip(norm_questions, checker_results) if cond]
    reader_responses = [q for q, cond in zip(reader_responses, checker_results) if cond]
    contexts = [q for q, cond in zip(contexts, checker_results) if cond]

    timer_prompts = []
    new_questions = []
    new_answers = []
    for q, nq, c, a_s in zip(questions, norm_questions, contexts, reader_responses):
        for a in a_s:
            a = a.replace('\n','').strip()
            new_questions.append(q)
            new_answers.append(a)
            timer_prompt = timer(nq, c, a)
            timer_prompts.append(timer_prompt)

    timer_responses = call_pipeline(args, timer_prompts, 100)
    # for x,y in zip(timer_prompts, timer_responses):
    #     print('\n==\n')
    #     print(x.split('Now your question, context paragraph and answer are as follows.')[-1])
    #     print(y)
    # import ipdb; ipdb.set_trace()

    # for "when" question, no need to do timer
    # ensure answer has a year

    for j in range(len(timer_responses)):
        if new_questions[j].lower().startswith('when'):
            try:
                eval(timer_responses[j])
            except Exception as e:
                ys = year_identifier(new_answers[j])
                if len(ys)>1:
                    print('>1')
                    print('{"'+new_answers[j]+'": {"start_year": '+str(min(ys))+', "start_month": 0, "end_year": '+str(max(ys))+', "end_month": 0}'+'}')
                    timer_responses[j] = '{"'+new_answers[j]+'": {"start_year": '+str(min(ys))+', "start_month": 0, "end_year": '+str(max(ys))+', "end_month": 0}'+'}'
                elif len(ys)==1:
                    print('=1')
                    print('{"'+new_answers[j]+'": {"start_year": '+str(ys[0])+', "start_month": 0, "end_year": '+str(ys[0])+', "end_month": 0}'+'}')
                    timer_responses[j] = '{"'+new_answers[j]+'": {"start_year": '+str(ys[0])+', "start_month": 0, "end_year": '+str(ys[0])+', "end_month": 0}'+'}'
                import ipdb; ipdb.set_trace()

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


    to_save=[]
    for k, ex in enumerate(examples):
        if ex['time_relation'] == '':
            continue

        question = ex['question']
        normalized_question = ex['normalized_question']
        gold_evidences = ex['gold_evidences']
        answer_dict = question_answer_map[question] if question in question_answer_map else {}

        # if question != "When was the last time the Dodgers played the Yankees in the World Series after 1978?":
        #     continue

        print('\n------\n', k,' ',question,'\n------\n') 
        
        print('\nanswer dict')
        print(answer_dict)

        # import ipdb; ipdb.set_trace()


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
        else:
            # if question.lower().startswith('when'):
            #     rag_pred = ans_list[0].values()['start_year']
            #     print(question, ' ', rag_pred)
            # else:
            rag_pred = next(iter(ans_list[0]))
            # tmp = str(ans_list[0][rag_pred]['start_year'])
            # if tmp in rag_pred:
            #     rag_pred = tmp
        rag_pred = rag_pred.replace('\n','').strip()
        
        ex['rag_pred'] = rag_pred
        ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
        ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
        print(rag_pred, ex['answers'], ex['rag_acc'])
        


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


        to_save.append(result)


    # to_save_df = pd.DataFrame(to_save)
    # retriever_name = args.retriever_output.split('/')[-1].split('_outputs')[0]
    # result_name = f'./answered/{retriever_name}_top{args.ctx_topk}_{args.llm_name}_results.csv'
    # to_save_df.to_csv(result_name, index=False, encoding='utf-8')
    # print(f"Saved as {result_name}")


    eval_reader(to_save, False, subset='situatedqa', metric='acc')
    eval_reader(to_save, False, subset='situatedqa', metric='f1')








if __name__ == "__main__":
    main()