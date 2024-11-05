import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse
from copy import deepcopy
# from vllm import LLM, SamplingParams
 
from temp_eval import normalize

# def checker(question, context):
#     prompt = f"""Consider the question and context paragraph below:

# <Context>
# {context}
# </Context>
# <Question>
# {question}
# </Question>

# Does the context provide an answer to the question?

# First, read the context and respond with either "Yes" or "No". Then, briefly explain your reasoning.

# <Response>
# """
#     return prompt

def checker(question, context):
    prompt = f"""You will be given a context paragraph and a question. As an assistant, your task is decide whether the context contains the answer to the question.
Requirements are follows:
- First read the paragraph after <Context> and question after <Question> carefully.
- Then you should think step by step and give your thought after <Thought>.
- Finally, write the response by "Yes" or "No" after <Response>.

<Context>
Petronas Towers | From 1996 to 2004, they were officially designated as the tallest buildings in the world until they were surpassed by the completion of Taipei 101. The Petronas Towers remain the world's tallest twin skyscrapers, surpassing the World Trade Center towers in New York City, and were the tallest buildings in Malaysia until 2019, when they were surpassed by The Exchange 106.
</Context>
<Question>
Tallest building in the world
</Question>
<Thought>
The context paragraph talks about the Petronas Towers. The context paragraph states that Petronas Towers were officially designated as the tallest buildings in the world from 1996 to 2004. And the Taipei 101 became the the tallest building in the world after 2004. This context paragraph contains two answers to the question. Therefore, the response is "Yes". 
</Thought>
<Response>
Yes
</Response>

There are some examples for you to refer to:
<Context>
Petronas Towers | The Petronas Towers (Malay: Menara Berkembar Petronas), also known as the Petronas Twin Towers and colloquially the KLCC Twin Towers, are an interlinked pair of 88-storey supertall skyscrapers in Kuala Lumpur, Malaysia, standing at 451.9 metres (1,483 feet).
</Context>
<Question>
Tallest building in the world
</Question>
<Thought>
The context paragraph talks about the Petronas Towers and their height of 451.9 metres (1,483 feet). However, it does not state the Petronas Towers is the tallest building in the world. The context paragraph does not tell which building is the tallest in the world. Therefore, the response is "No". 
</Thought>
<Response>
No
</Response>

<Context>
List of 20th-century religious leaders Church of England | Formal leadership: Supreme Governor of the Church of England (complete list) – ; Victoria, Supreme Governor (1837–1901) ; Edward VII, Supreme Governor (1901–1910) ; George V, Supreme Governor (1910–1936) ; Cosmo Gordon Lang, Archbishop of Canterbury (1928–1942) ; William Temple, Archbishop of Canterbury (1942–1944) ; 
</Context>
<Question>
Who is the head of the Church in England
</Question>
<Thought>
The context paragraph talks about the 20th-century religious leaders Church of England. In this list, it states the names of Supreme Governor of the Church of England, which is the head of the Church in England. This context contains the answers for the head of the Church in England: Victoria, Edward VII, and George V. Therefore, the response is "Yes". 
</Thought>
<Response>
Yes
</Response>

Now your context paragraph and question are as follows.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Thought>
"""
    return prompt


def reader(question, title, text):

    prompt = f"""You will be given a context paragraph and a question. First read the context paragraph carefully. Only based on this context information, write the answers in standalone sentences for the question accurately. Sentences should start after <Answer> and end with </Answer>.
Requirements are follows:
- Only include one anwer in one sentence per line.
- Each sentence should also include the date corresponding to the answer if it is mentioned or can be inferred.
- If the context knowledge contains no answer to the question, write "None".

There are some examples for you to refer to:
<Context>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995. They have also won the champions in 1996-1997, 1999, 2000-01. 
</Context>
<Question>
When did the Houston Rockets win the NBA championship
</Question>
<Answer>
- Houston Rockets won the NBA championship in 1994.
- Houston Rockets won the NBA championship in 1995.
- Houston Rockets won the NBA championship in 1996-1997.
- Houston Rockets won the NBA championship in 1999.
- Houston Rockets won the NBA championship in 2000-01.
</Answer>

<Context>
List of presidents of India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India
</Question>
<Answer>
- Neelam Sanjiva Reddy served the sixth President of India from 1977.
- K. R. Narayanan served as the President of India from 1997 until 2002.
- Droupadi Murmu served as the 15th President from 2022.
</Answer>

<Context>
Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
- The worldwide box office of Jurassic movie - The Lost World: Jurassic Park was $618.6 million in 1997.
- The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
</Answer>

<Context>
1980 Summer Olympics | Hence, the selection process for the 1984 Summer Olympics consisted of a single finalized bid from Los Angeles, which the International Olympic Committee (IOC) accepted in 1976. Los Angeles was awarded the selection officially on May 18, 1978 for the 1984 Summer Olympics.
</Context>
<Question>
When has the United States hosted Summer Olympics
</Question>
<Answer>
- The United States has hosted 1984 Summer Olympics.
</Answer>

<Context>
Oliver Bulleid | A brief period working for the Board of Trade followed from 1910, arranging exhibitions in Brussels, Paris and Turin. He was able to travel widely in Europe, later including a trip with Nigel Gresley, William Stanier and Frederick Hawksworth, to Belgium, in 1934, to see a metre-gauge bogie locomotive. In December 1912, he rejoined the GNR as Personal Assistant to Nigel Gresley, the new CME. Gresley was only six years Bulleid's senior.
</Context>
<Question>
Oliver Bulleid was an employee for whom
</Question>
<Answer>
- Oliver Bulleid was an employee for the Board of Trade from 1910.
- Oliver Bulleid was an employee for the GNR from December 1912.
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


def timer(question, answer):
    prompt = f"""You will be given one question and one context sentence. Your task is to find the answer and corresponding date. Your response should be in a python dict object.
- The date should be parsed into a python dict format with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date such as an event, write the same start and end time.
- If the answer applies "from" a specific date such as job and political positions, write this date as the start time and write "0" for the end time.
- If the answer applies "until" a specific date such as job and political positions, write this date as the end time and write "0" for the start time.

There are some examples for you to refer to:
<Context>
Neelam Sanjiva Reddy served the sixth President of India from 1977.
</Context>
<Question>
Who served as President of India
</Question>
<Answer>
{{"Neelam Sanjiva Reddy": {{"start_year": 1977, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Answer>

<Context>
K. R. Narayanan served as the President of India from 1997 until 2002.
</Context>
<Question>
Who served as President of India
</Question>
<Answer>
{{"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}}}}
</Answer>

<Context>
The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
{{"$914 million": {{"start_year": 1993, "start_month": 6, "end_year": 1993, "end_month": 6}}}}
</Answer>

<Context>
Houston Rockets won the NBA championship in 1996-1997.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
{{"1996-1997": {{"start_year": 1996, "start_month": 0, "end_year": 1997, "end_month": 0}}}}
</Answer>

<Context>
Houston Rockets won the NBA championship in 2000-01.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
{{"2000-01": {{"start_year": 2000, "start_month": 0, "end_year": 2001, "end_month": 0}}}}
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
</Context>
<Question>
When has the United States hosted Summer Olympics
</Question>
<Answer>
{{"1984": {{"start_year": 1984, "start_month": 0, "end_year": 1984, "end_month": 0}}}}
</Answer>

<Context>
Oliver Bulleid was an employee for the Board of Trade from 1910.
</Context>
<Question>
Oliver Bulleid was an employee for whom
</Question>
<Answer>
{{"Board of Trade": {{"start_year": 1910, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Answer>

Now your context sentence and question are as follows.
<Context>
{answer}
</Context>
<Question>
{question}
</Question>
<Answer>
"""
    return prompt


def combiner(question, contexts):
    prompt = f"""You will be given a context paragraph and a question. As an assistant, your task is to answer the question only based on the information from the context. You should first think step by step about the question and give your thought and then answer the <Question>. Your thought should be after <Thought>. Your answer should be after <Answer>. If there is no answer in the context, response "None".
    
There are some examples for you to refer to:

<Context>
England hosted the World Cup and went on to win the tournament in 1966, defeating West Germany 4-2 in the final.
England reached the World Cup semi-finals in 1990.
England made it to the World Cup semi-finals in 2018.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>
The question asks about the time when England last get to the semi final of a World Cup before 2019. The answer should be a date. Based on the context, 2018 is the last time when England got to the World Cup semi-finals. 2018 is before 2019. Therefore, the answer is 2018.
</Thought>
<Answer>
2018
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
The United States has hosted Summer Olympics in 1984.
The United States has hosted Summer Olympics in 1996.
</Context>
<Question>
How many times had the United States hosted Summer Olympics before 2000?
</Question>
<Thought>
The question asks about the number of times that the United States had hosted Summer Olympics before 2000. The answer should be an integer. Based on the context, the United States has hosted Summer Olympics twice in 1984 and 1996. 1984 and 1996 are before 2000. Therefore, the answer is 2.
</Thought>
<Answer>
2
</Answer>

<Context>
Neelam Sanjiva Reddy served the sixth President of India from 1977.
K. R. Narayanan served as the President of India from 1997 until 2002.
</Context>
<Question>
Who is the President of India on Jan 10, 1998?
</Question>
<Thought>
The question asks about the person of the President of India on Jan 10, 1998. The answer should be a person's name. Based on the context, K. R. Narayanan served as the President of India from 1997 until 2002. Jan 10, 1998 is between 1997 and 2002. Therefore, the answer is K. R. Narayanan.
</Thought>
<Answer>
K. R. Narayanan
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
The United States has hosted Summer Olympics in 1984.
The United States has hosted Summer Olympics in 1996.
</Context>
<Question>
When is the last Summer Olympics that the United States hosted as of 2000?
</Question>
<Thought>
The question asks about the time of the last Summer Olympics that the United States hosted as of 2000. The answer should be a date. Based on the context, the last time when the United States hosted Summer Olympics is 1996. 1996 is not later than 2000. Therefore, the answer is 1996.
</Thought>
<Answer>
1996
</Answer>

<Context>
The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
The worldwide box office of Jurassic movie - The Lost World: Jurassic Park was $618.6 million in 1997.
</Context>
<Question>
What was the worldwide box office of the first Jurassic movie after 1990?
</Question>
<Thought>
The question asks about the worldwide box office of the first Jurassic movie after 1990. The answer should be a monetary value. Based on the context, the first Jurassic movie is Jurassic Park premiered on June 9, 1993. 1993 is after 1990. The worldwide box office of Jurassic Park was $914 million. Therefore, the answer is $914 million.
</Thought>
<Answer>
$914 million
</Answer>

<Context>
Oliver Bulleid was an employee for the Board of Trade from 1910.
Oliver Bulleid was an employee for the GNR from December 1912.
</Context>
<Question>
Oliver Bulleid was an employee for whom between 1911 and 1912?
</Question>
<Thought>
The question asks about the employer of Oliver Bulleid between 1911 and 1912. The answer should be a name of company or organization. Based on the context, Oliver Bulleid started to work for the Board of Trade from 1910 and GNR from December 1912. 1911 and 1912 are after 1910 and before December 1912. Oliver Bulleid worked for the Board of Trade between 1911 and 1912. Therefore, the answer is Board of Trade.
</Thought>
<Answer>
Board of Trade
</Answer>

Now your context paragraph and question are as follows.

<Context>
{contexts}
</Context>
<Question>
{question}
</Question>
<Thought>
"""
    return prompt


# <Context>
# The worldwide box office of Jurassic movie - The Lost World: Jurassic Park was $618.6 million in 1997.
# The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
# </Context>
# <Question>
# What was the worldwide box office of the last Jurassic movie before 2000?
# </Question>
# <Answer>
# $618.6 million
# </Answer>

# <Context>
# Houston Rockets won the NBA championship in 1994.
# Houston Rockets won the NBA championship in 1995.
# Houston Rockets won the NBA championship in 1996-1997.
# </Context>
# <Question>
# When was the first time Houston Rockets won the NBA championship between 1993 and Dec 1997?
# </Question>
# <Answer>
# 1994
# </Answer>

# <Context>
# Germany was the team to win the 2014 World Cup.
# </Context>
# <Question>
# Which was the first team to win the World Cup between 2012 and 2018?
# </Question>
# <Answer>
# Germany
# </Answer>

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
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=10)
    parser.add_argument('--param-pred', type=bool, default=True)
    parser.add_argument('--param-cot', type=bool, default=False)
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--save-note', type=str, default=None)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    # parser.add_argument('--ctx-key-s2', type=str, default='snt_hybrid_rank')
    # parser.add_argument('--ctx-key-s2', type=str, default='reranker_ctxs')
    parser.add_argument('--reader', type=str, default='llama', choices=['llama', 'timo', 'timellama'])
    parser.add_argument('--paradigm', type=str, default='fusion', choices=['fusion', 'concat'])


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
            args.llm = LLM(args.l, tensor_parallel_size=2, dtype='float16', max_model_len=4096)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    if args.max_examples:
        # examples = examples[:min(len(examples),args.max_examples)]
        examples = examples[-args.max_examples:]

    # examples = examples[100:200]
    # x = "Who is in charge of the Minister of Personnel, Public Grievances, and Pensions since 27 May 2014?"
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

            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                context = f"{ctx['title']} | {ctx['text']}"
                checker_prompt = checker(normalized_question, context)
                checker_prompts.append(checker_prompt)

                reader_prompt = reader(normalized_question, ctx['title'], ctx['text'])
                reader_prompts.append(reader_prompt)
        
        checker_responses = call_pipeline(args, checker_prompts, 500)
        checker_results = ['yes' in res.lower() for res in checker_responses]
        print('started reader')
        reader_responses = call_pipeline(args, reader_prompts, 500)


        for x, y, z in zip(reader_prompts, reader_responses, checker_responses):
            print('\n==\n')
            print(x.split('Now your context paragraph and question are as follows.')[-1])
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
                    reader_ans += reader_response
            reader_ans = list(set(reader_ans))
            for reader_a in reader_ans:
                sub_examples.append([question, normalized_question, reader_a])
                timer_prompts.append(timer(normalized_question, reader_a))
        if len(sub_examples)>0:
            timer_responses = call_pipeline(args, timer_prompts, 100)
        else:
            timer_responses = []

        question_result_map = {}
        for sub_ex, timer_r in zip(sub_examples, timer_responses):
            try:
                answer_dict = eval(timer_r)
                assert isinstance(answer_dict, dict)
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
            print('\nanswer dicts')
            print(result)
        
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
                # if ex['implicit_condition'] == 'last':
                #     filtered_result = sorted(filtered_result, key=lambda x: (list(x[-1].values())[0]['start_year'], list(x[-1].values())[0]['end_year'], list(x[-1].values())[0]['start_month']), reverse=True)
                # elif ex['implicit_condition'] == 'first':
                filtered_result = sorted(filtered_result, key=lambda x: (list(x[-1].values())[0]['start_year'], list(x[-1].values())[0]['start_month']), reverse=False)
                # else:
                #     if ex['time_relation_type']=='between':
                #         filtered_result = sorted(filtered_result, key=lambda x: (abs(list(x[-1].values())[0]['start_year']-ex['years'][0]), abs(list(x[-1].values())[0]['end_year']-ex['years'][1])), reverse=False)
                #     else:
                #         filtered_result = sorted(filtered_result, key=lambda x: abs(list(x[-1].values())[0]['start_year']-ex['years'][0]), reverse=False)
            
            new_questions.append(question)
            contexts = '\n'.join([tp[2].replace(' first','').replace(' last','') for tp in filtered_result])
            combiner_prompt = combiner(question, contexts)
            combiner_prompts.append(combiner_prompt)
        
        if len(combiner_prompts)>0:
            combiner_responses = call_pipeline(args, combiner_prompts, 500)
            combiner_responses = [res.replace('\n','').strip() for res in combiner_responses]
        else:
            combiner_responses = []

        for x,y in zip(combiner_prompts, combiner_responses):
            print(x.split('Now your context paragraph and question are as follows.')[-1])
            print(y)

        question_answer_map = {q: a for q, a in zip(new_questions, combiner_responses) if len(a)>0}
        
        for k, ex in enumerate(examples):
            question = ex['question']
            find_flg = True
            if question in question_answer_map:
                rag_pred = question_answer_map[question]
                if any([item in rag_pred.lower() for item in ['unknown', 'none', 'not provided', "do not know"]]):
                    find_flg = False
            else:
                find_flg = False

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
                rag_pred = call_pipeline(args, [prompt], 100)[0]

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