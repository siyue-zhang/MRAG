import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

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


def checker(question, context):
    prompt = f"""Does the context paragraph contain the answer to the question? Response Yes or No.
There are some examples for you to refer to:
<Context>
Grigsby volunteered for World War II in 1942 and served in the Army.
</Context>
<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Response>
Yes
</Response>

<Context>
Lakers won the NBA champion in 2007.
</Context>
<Question>
The time when the Huston Rockets won the NBA champion
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

def all_in_one_reader(question, title, text):
    prompt = f"""You will be given a question and a context paragraph. Your task is to extract the answer and the corresponding date from the context.
- The result should be in the python dict format: the extracted answer is the dict key and the corresponding date is the dict value.
- Ensure the dict key is the answer to the question such as a name, date, organization, etc.
- The date should be parsed into a python dict object with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date, write the same start and end time.
- If the answer applies from a specific date, write this date as the start time and write "0" for the end time.
- If the answer applies until a specific date, write this date as the end time and write "0" for the start time.
- Write "0" if the date data is not available.

There are some examples for you to refer to:
<Context>:
Houston Rockets | The team have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in May 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
{{
"1994": {{"start_year": 1994, "start_month": 0, "end_year": 1994, "end_month": 0}},
"May 1995": {{"start_year": 1995, "start_month": 5, "end_year": 1995, "end_month": 5}}
}}
</Answer>

<Context>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president. She served until 2024.
</Context>
<Question>
Who served as President of India
</Question>
<Answer>
{{
"Neelam Sanjiva Reddy": {{"start_year": 1977, "start_month": 0, "end_year": 0, "end_month": 0}},
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}},
"Droupadi Murmu": {{"start_year": 0, "start_month": 0, "end_year": 2024, "end_month": 0}}
}}
</Answer>

<Context>
The Lost World: Jurassic Park | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America.
</Context>
<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Answer>
{{
"$618.6 million": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}
}}
</Answer>


<Context>
J. Eugene Grigsby | Grigsby volunteered for World War II in 1942 and served in the Army. Starting in 1946 Grigsby served as the Founder and Chair of the Art Department at Carver High School for eight years.
</Context>
<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Answer>
{{
"Army": {{"start_year": 1942, "start_month": 0, "end_year": 0, "end_month": 0}},
"Carver High School": {{"start_year": 1946, "start_month": 0, "end_year": 1954, "end_month": 0}}
}}
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

# - The Houston Rockets claimed their second NBA championship in 1995.
# - Each sentence should contain only one answer with corresponding date.
# <Context>
# Madison Marsh | Madison Isabella Marsh (born August 2, 2001) is an American beauty pageant titleholder who was crowned Miss America 2024. She had previously been crowned Miss Colorado 2023.
# </Context>
# <Question>
# When was the time Miss Colorado won Miss America
# </Question>
# <Answer>
# - Madison Marsh, who was crowned Miss Colorado 2023, was crowned Miss America in 2024.
# </Answer>

# <Context>
# 2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
# </Context>
# <Question>
# Who won the Grand National
# </Question>
# <Answer>
# None
# </Answer>

def reader(question, title, text):
    prompt = f"""You will be given a context paragraph and a question. Your task is to write independent sentences to answer the question based on the context paragraph. 
Requirements are follows:
- Write one standalone sentence per line with specific subjects, objects, relations, and actions.
- If there is no answer to the question from the context paragraph, respond with "None". 

There are some examples for you to refer to:
<Context>:
Houston Rockets | The team have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Answer>
- The Houston Rockets won the NBA championship in 1994 and 1995.
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
- The movie "The Lost World: Jurassic Park" grossed a total of $618.6 million at the worldwide box office in 1997.
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
    prompt = f"""You will be given a question and a sentence. Your task is to extract the answer and the corresponding date from the sentence.
- The result should be in the python dict format: the extracted answer is the dict key and the corresponding date is the dict value.
- Ensure the dict key is the answer to the question such as a name, date, organization, etc.
- The date should be parsed into a python dict object with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date, write the same start and end time.
- If the answer applies from a specific date, write this date as the start time and write "0" for the end time.
- If the answer applies until a specific date, write this date as the end time and write "0" for the start time.
- Write "0" if the date data is not available.

There are some examples for you to refer to:
<Question>
Who served as President of India
</Question>
<Sentence>
K. R. Narayanan served as the President of India from 1997 until 2002, Droupadi Murmu served until 2024.
</Sentence>
<Answer>
{{
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}},
"Droupadi Murmu": {{"start_year": 0, "start_month": 0, "end_year": 2024, "end_month": 0}}
}}
</Answer>

<Question>
What was the worldwide box office of Jurassic movie
</Question>
<Sentence>
The movie "The Lost World: Jurassic Park" grossed a total of $618.6 million at the worldwide box office in 1997.
</Sentence>
<Answer>
{{
"$618.6 million": {{"start_year": 1997, "start_month": 0, "end_year": 1997, "end_month": 0}}
}}
</Answer>

<Question>
When was the time the Houston Rockets win the NBA championship
</Question>
<Sentence>
The Houston Rockets won the NBA championship in 1994 and May 1995.
</Sentence>
<Answer>
{{
"1994": {{"start_year": 1994, "start_month": 0, "end_year": 1994, "end_month": 0}},
"May 1995": {{"start_year": 1995, "start_month": 5, "end_year": 1995, "end_month": 5}}
}}
</Answer>

<Question>
Who serve as President of India
</Question>
<Sentence>
Neelam Sanjiva Reddy served as the sixth President of India from Dec 1977, K. R. Narayanan - President of India (1997-98).
</Sentence>
<Answer>
{{
"Neelam Sanjiva": {{"start_year": 1977, "start_month": 12, "end_year": 0, "end_month": 0}},
"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 1998, "end_month": 0}},
}}
</Answer>

<Question>
J. Eugene Grigsby was an employee for whom
</Question>
<Sentence>
Grigsby volunteered for World War II in 1942 and served in the Army. Starting in 1946 Grigsby served as the Founder and Chair of the Art Department at Carver High School for eight years.
</Sentence>
<Answer>
{{
"Army": {{"start_year": 1942, "start_month": 0, "end_year": 0, "end_month": 0}},
"Carver High School": {{"start_year": 1946, "start_month": 0, "end_year": 1954, "end_month": 0}}
}}
</Answer>

Now your context sentence and question are as follows.
<Question>
{question}
</Question>
<Sentence>
{sentence}
</Sentence>
<Answer>
"""
    return prompt


# def formatter(question, answer):
    
#     prompt = f"""You will be given a question and a sentence. Your task is to first determine if the sentence can clearly answer the question.
# If the sentence does not clearly answer the question, response with "None".
# If the sentence can answer the question, then
# - Write the answer in the first line. 
# - Write the corresponding date (i.e., year, month, day) in the second line in python dictionary format. 
# - Each date should be parsed into a dict object with keys ("start_year", "start_month", "end_year", "end_month"). If data is not available, write "0".

# There are some examples for you to refer to:
# <Question>
# Who serve as President of India
# </Question>
# <Sentence>
# K. R. Narayanan served as the President of India from 1997 until 2002.
# </Sentence>
# <Answer>
# K. R. Narayanan
# {{"start_year": "1997", "start_month": "0", "end_year": "2002", "end_month": "0"}}
# </Answer>

# <Question>
# Who serve as President of India
# </Question>
# <Sentence>
# K. R. Narayanan served as the President of India.
# </Sentence>
# <Answer>
# K. R. Narayanan
# {{"start_year": "0", "start_month": "0", "end_year": "0", "end_month": "0"}}
# </Answer>

# <Question>
# Who serve as President of India
# </Question>
# <Sentence>
# Droupadi Murmu served as the President of India from Dec 2022.
# </Sentence>
# <Answer>
# Droupadi Murmu
# {{"start_year": "2022", "start_month": "12", "end_year": "0", "end_month": "0"}}
# </Answer>

# <Question>
# Who serve as President of India
# </Question>
# <Sentence>
# Droupadi Murmu served as the President of India unitil 2022.
# </Sentence>
# <Answer>
# Droupadi Murmu
# {{"start_year": "0", "start_month": "0", "end_year": "2022", "end_month": "0"}}
# </Answer>

# <Question>
# When did the Houston Rockets win the NBA championship
# </Question>
# <Sentence>
# The Houston Rockets have won the NBA championship in 1994.
# </Sentence>
# <Answer>
# 1994
# {{"start_year": "1994", "start_month": "0", "end_year": "1994", "end_month": "0"}}
# </Answer>

# <Question>
# When did the Houston Rockets win the NBA championship
# </Question>
# <Sentence>
# The Houston Rockets have won the NBA championship on June 2, 1995.
# </Sentence>
# <Answer>
# June 2, 1995
# {{"start_year": "1995", "start_month": "6", "end_year": "1995", "end_month": "6"}}
# </Answer>

# <Question>
# When did the Houston Rockets win the NBA championship
# </Question>
# <Sentence>
# The Houston Rockets won the NBA championship on 10 May 1980.
# </Sentence>
# <Answer>
# 10 May 1980
# {{"start_year": "1980", "start_month": "5", "end_year": "1980", "end_month": "5"}}
# </Answer>

# <Question>
# When did the Houston Rockets win the NBA championship
# </Question>
# <Sentence>
# The Lakers have won the NBA championship on June 2, 1995.
# </Sentence>
# <Answer>
# None
# </Answer>

# <Question>
# What was the worldwide box office of Jurassic movie
# </Question>
# <Sentence>
# The Lost World: Jurassic Park has a total of $618.6 million worldwide box office in 1997.
# </Sentence>
# <Answer>
# $618.6 million
# {{"start_year": "1997", "start_month": "0", "end_year": "1997", "end_month": "0"}}
# </Answer>

# <Question>
# {question}
# </Question>
# <Sentence>
# {answer}
# </Sentence>
# <Answer>
# """
     
#     return prompt



# query focused summarizer
# def get_QFS_prompt(question, title, text):
#     prompt = f"""You are given a paragraph and a specific question. Your goal is to summarize the paragraph after <Context> in complete sentences by answering the given question. If dates are mentioned in the paragraph, include them in your answer. If the question cannot be answered based on the paragraph, respond with "None." Ensure that the response is concise and directly addresses the question.
# There are some examples for you to refer to:
# <Context>:
# Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
# <Question>:
# When did the Houston Rockets win the NBA championship
# <Summarization>:
# The Houston Rockets won the NBA championship in 1994 and 1995.
# <Context>:
# 2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
# <Question>:
# Who won the Grand National
# <Summarization>:
# None
# <Context>:
# India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
# <Question>:
# Who serve as President of India
# <Summarization>:
# Neelam Sanjiva Reddy served as President in 1977, K. R. Narayanan in 1997, and Droupadi Murmu in 2022.

# Now your question and paragraph are as follows.
# <Context>:
# {title} | {text}
# <Question>:
# {question}
# <Summarization>:
# """
#     return prompt

# def c_prompt(query, texts):

#     prompt=f"""Answer the given question, you can refer to the document provided.
# As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer>.
# The given knowledge will be after the <Context> tag. You can refer to the knowledge to answer the question.
# If the knowledge does not contain the answer, answer the question directly.
# There are some examples for you to refer to:
# <Context>:
# hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

# The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

# They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
# <Question>:
# When did England last get to the semi final of a World Cup before 2019?
# <Answer>:
# 2018
# <Context>:
# For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

# For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
# <Question>:
# Who sang the national anthem in the last Super Bowl as of 2021?
# <Answer>:
# Eric Church and Jazmine Sullivan
# <Context>:
# Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

# Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
# <Question>:
# Where was the last Rugby World Cup held between 2007 and 2016?
# <Answer>:
# England

# Now your question and reference knowledge are as follows.
# <Context>:
# {texts}
# <Question>:
# {query}
# <Answer>:
# """
#     return prompt


def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=10)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument('--param-cot', type=bool, default=True)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--ctx-key-s2', type=str, default='snt_hybrid_rank')
    # parser.add_argument('--ctx-key-s2', type=str, default='reranker_ctxs')
    parser.add_argument('--reader', type=str, default='timellama', choices=['rag', 'metriever', 'timo', 'timellama', 'extract_code'], help="Choose a reader option")
    parser.add_argument('--temporal-filter', type=bool, default=False)

    args = parser.parse_args()
    args.l = llm_names(args.llm, instruct=True)
    args.llm_name = deepcopy(args.llm)

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

    # if args.max_examples:
        # examples = examples[:min(len(examples),args.max_examples)]
    examples = examples[381:382]

    
    to_save=[]
    for k, ex in enumerate(examples):
        if ex['time_relation'] == '':
            continue

        question = ex['question']
        question = question.replace('annd', 'and')
        ex['time_relation'] = ex['time_relation'].strip()

        
        # temp
        years = ex['years'] # question dates
        time_relation = ex['time_relation'].strip().lower()
        implicit_condition = ex['implicit_condition']
        if time_relation in ['before','as of','by','until']:
            time_relation_type = 'before'
        elif time_relation == 'from':
            if len(years)==1:
                time_relation_type = 'after'
            else:
                time_relation_type = 'between'
        elif time_relation == 'since':
            time_relation_type = 'after'
        elif time_relation in ['after','between']:
            time_relation_type = time_relation
        else:
            time_relation_type = 'other'
        ex['time_relation_type'] = time_relation_type
        
        # if question != "When was the last time the Dodgers played the Yankees in the World Series after 1978?":
        #     continue
        print('\n------\n',question,'\n------\n') 


        date = ''
        parts = question.split(ex['time_relation'])
        date = parts[-1]
        ex['date'] = date

        def find_month(w):
            w = w.lower()
            month = []
            for m in month_to_number:
                if m in w:
                    month.append(month_to_number[m])
                    break
            for m in short_month_to_number:
                if m in w:
                    month.append(short_month_to_number[m])
                    break
            if len(month)>0:
                return month[0]
            else:
                return None
        

        months = []

        def append_month(month_str):
            m = find_month(month_str)
            months.append(m if m else 0)

        if ex['time_relation_type'] == 'between':
            delimiters = ['and', 'to', 'until']
            d_index = [d in date for d in delimiters]
            assert any(d_index)
            delimiter = delimiters[d_index.index(True)]
            tmp = date.split(delimiter)
            for w in tmp:
                append_month(w.strip())
        else:
            append_month(date.strip())

        ex['months'] = months
        print('months ', months)



            # new_texts = []
        print('\n------\n',question,'\n------\n') 

        ans_list = []
        # errors = []
        rel_events = []
        for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
            normalized_question = ex['normalized_question']
            prompt = checker(normalized_question, ctx['title']+' | '+ctx['text'])
            responses = call_pipeline(args, [prompt], 10)
            response = responses[0]
            if response[:3].lower()=='yes':
                prompt = reader(normalized_question+' and when', ctx['title'], ctx['text'])
                responses = call_pipeline(args, [prompt], 200)
                sentence_list = responses[0]
            else:
                sentence_list = []

            rel_events += sentence_list

            # print('\n', ctx['title'], ctx['text'])
            # print(sentence_list)
        
        rel_events = ' '.join(rel_events)
        prompt = formatter(normalized_question, rel_events)
        responses = call_pipeline(args, [prompt], 500)
        response = responses[0]
        print(rel_events)
        print(response)
        import ipdb; ipdb.set_trace()

            # if isinstance(sentence_list, list):
            #     for sentence in sentence_list:
            #         prompt = checker(normalized_question, sentence)
            #         print('vv\n', prompt)
            #         responses = call_pipeline(args, [prompt], 10)
            #         response = responses[0]
            #         print(response)

            #         if response[:3].lower()=='yes':
            #             prompt = formatter(normalized_question, sentence)
            #             responses = call_pipeline(args, [prompt])
            #             answer_dict = responses[0]
            #             print('formatter response:\n', answer_dict)
            #             try:
            #                 # answer_dict = answer_dict.split('\n')
            #                 # answer_dict = [s for s in answer_dict if len(s)>0]
            #                 # tmp = eval(answer_dict[1])
            #                 # answer_dict = {answer_dict[0]: tmp}
            #                 answer_dict = eval(answer_dict)
            #                 print('parsing success:\n-->', answer_dict)
            #             except Exception as e:
            #                 print('parsing failed.\n')
            #                 pass

            #             if isinstance(answer_dict, dict):
            #                 for ans in answer_dict:
            #                     tmp = {ans: answer_dict[ans]}
            #                     print('find one answer: ', tmp)
            #                     # import ipdb; ipdb.set_trace()
            #                     if tmp not in ans_list:
            #                         ans_list.append(tmp)

            #                     # k = next(iter(answer_dict))
            #                     # if any([ss in sentence.lower() for ss in [f'in {k}', f'on {k}']]):
            #                     #     print('[[[[]]]]',answer_dict)
            #                     #     if not isinstance(answer_dict[k], dict):
            #                     #         continue
            #                     #     answer_dict[k]['start_year'] = answer_dict[k]['end_year']
            #                     #     answer_dict[k]['start_month'] = answer_dict[k]['end_month']
            #                     # if answer_dict not in ans_list:
            #                     #     print(answer_dict)
            #                     #     ans_list.append(answer_dict)
            #                 # else:
            #                 #     errors.append(answer_dict)
            #         else:
            #             print('the answer sentence is irrelevant.\n', sentence)
        print('finish all contexts.')
        # print(ans_list)
        # import ipdb; ipdb.set_trace()
        # for ans_date in ans_list:
        #     ans = next(iter(ans_date))
        #     ans_date[ans]['start_year'] = int(ans_date[ans]['start_year'])
        #     ans_date[ans]['end_year'] = int(ans_date[ans]['end_year'])
        #     ans_date[ans]['start_month'] = int(ans_date[ans]['start_month'])
        #     ans_date[ans]['end_month'] = int(ans_date[ans]['end_month'])


        # import ipdb; ipdb.set_trace()

        tmp = []
        if ex['time_relation_type']=='before':
            q_year = ex['years'][0]
            q_month = ex['months'][0] if sum(ex['months'])>0 else None

            for ans_date in ans_list:
                ans = next(iter(ans_date))
                start_year = ans_date[ans]['start_year']
                start_month = ans_date[ans]['start_month']

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
                    tmp.append(ans_date)

        elif ex['time_relation_type']=='after':
            q_year = ex['years'][0]
            q_month = ex['months'][0] if sum(ex['months'])>0 else None

            for ans_date in ans_list:
                ans = next(iter(ans_date))
                end_year = ans_date[ans]['end_year']
                end_month = ans_date[ans]['end_month']              
                
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
                    tmp.append(ans_date)

        elif ex['time_relation_type']=='between':
            q_year_s = ex['years'][0]
            # q_month_s = ex['months'][0]
            q_year_e = ex['years'][1]
            # q_month_e = ex['months'][1]
            for ans_date in ans_list:
                ans = next(iter(ans_date))
                start_year = ans_date[ans]['start_year']
                end_year = ans_date[ans]['end_year']
                
                append_flg = True
                if start_year>q_year_e:
                    append_flg = False
                if end_year>0 and end_year<q_year_s:
                    append_flg = False
                
                if append_flg:
                    tmp.append(ans_date)

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
            rag_pred = next(iter(ans_list[0]))
        print(rag_pred, ex['answers'])

        
        ex['rag_pred'] = rag_pred
        ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
        ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))

        

        to_save.append(ex)

    eval_reader(to_save, False, subset='situatedqa', metric='acc')
    eval_reader(to_save, False, subset='situatedqa', metric='f1')




            # import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    main()