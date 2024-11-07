import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# def call_pipeline(args, prompts):
#     sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
#     outputs = args.llm.generate(prompts, sampling_params)
#     # print('~~~')
#     # print(prompts[0],'\n<>')
#     # print(outputs[0].outputs[0].text)
#     # print('~~~')
#     responses = [output.outputs[0].text for output in outputs]
#     responses = [res.split('<Context>:')[0] if '<Context>:' in res else res for res in responses]
#     responses = [res.split('Note:')[0] if 'Note:' in res else res for res in responses]
#     responses = [res.split('<Question>:')[0] if '<Question>:' in res else res for res in responses]
#     responses = [res.split('<Summarization>:')[0] if '<Summarization>:' in res else res for res in responses]
#     responses = [res.split('\n')[0].strip() for res in responses]
#     # responses = [res.replace('\n','').strip() for res in responses]
#     return responses 

def decontext(title, text, snt):
    prompt = f"""You will be given a context paragraph and a core section of this context. Your task is to convert the core section into independent sentences.
Requirements are follows:
- Write one sentence in one line.
- Same events occur in different years should be included in one sentence.
- Different events should be included in separate sentences.
- Each sentence should stand alone with complete information from the context, such as the name and date.
- Only use the words "in", "from", and "until" for the date. Use "from until" instead of "from to".

There are some examples for you to refer to:
<Context>
India | India has been a federal republic since 1950, governed through a democratic parliamentary system. India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Section>
In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Section>
<Sentences>
- Neelam Sanjiva Reddy served as the President of India from 1977.
- K. R. Narayanan served as the President of India from 1997 until 2002.
- Droupadi Murmu served as the President of India from 2022.
</Sentences>
<Context>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Section>
Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic.
</Section>
<Sentences>
- The Houston Rockets won the NBA championship in 1994, 1995.
</Sentences>

Now your context paragraph and the core section are as follows.
<Context>
{title} | {text}
</Context>
<Section>
{snt}
</Section>
<Sentences>
"""
    return prompt




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

    if args.max_examples:
        # examples = examples[:min(len(examples),args.max_examples)]
        examples = examples[-args.max_examples:]

    

    # for k, ex in enumerate(examples):
    #     if ex['time_relation'] != '':
    #         snts = ex['top_snt_id']
    #         texts = []
    #         for ctx in ex['snt_hybrid_rank']:
    #             ctx_id = ctx['id']
    #             title = ctx['title']
    #             ctx_snts = []
    #             text = title + ' | ' + ctx['text']
    #             while len(snts)>0 and snts[0][0] == ctx_id:
    #                 ctx_snts.append(snts.pop(0))
    #             for snt in ctx_snts:
    #                 snt = title.join(snt[1].split(title)[1:]).strip()
    #                 text = text.replace(snt, f"[[{snt}]]")
    #                 texts.append(text)
    #         prompt = c_bracket_prompt(ex['question'], '\n'.join(texts))
    #         responses = call_pipeline(args, [prompt])
    #         print(prompt, responses[0])
    #         import ipdb; ipdb.set_trace()

#             prompt=f'''Complete the sentence in the bracket in the context. Resolve the pronoun with specific name, infer the date and year.
# There are some examples for you to refer to:
# <Question>:
# When was the time the Dodgers played the Yankees in the World Series
# <Context>:
# 1941 World Series | [In 1947 the Yankees and the Dodgers would meet in the World Series for the second time and again play a dramatic Game 4 which was decided on a lead change with two outs in the ninth inning.] That time the Dodgers would be on the winning side to tie the series but would once again end up losing it. Ironically, in the 1947 game the Dodgers’ winning pitcher was none other than Hugh Casey – the Game 4 loser in 1941 – even though he pitched to only one batter.
# <Sentence>:
# The Dodgers played the Yankees in the World Series in 1947.
# Now your Question is
# <Question>:
# {ex['normalized_question']}
# <Context>:
# {text}
# <Sentence>:
# '''
    to_save=[]
    for k, ex in enumerate(examples):
        question = ex['question']
        if question != "When was the last time the Dodgers played the Yankees in the World Series before October 2, 2008?":
            continue
        print('\n------\n',question,'\n------\n') 


        if ex['time_relation'] != '':
            new_texts = []

            # for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
            #     normalized_question = ex['normalized_question']
            #     prompt = get_QFS_prompt(normalized_question, ctx['title'], ctx['text'])

            #     responses = call_pipeline(args, [prompt])
            #     response = responses[0].replace('\n','').strip()
            #     print('xxxxx')
            #     print(ctx['text'])
            #     print('--> ', response)
            #     if 'none' not in response.lower():
            #         print('before: ', response)
            #         if args.temporal_filter:
            #             summary_years = year_identifier(response)
            #             if summary_years:
            #                 def snt_temporal_filter(response, summary_years, y):
            #                     if response==None:
            #                         return None
            #                     if len(summary_years)>0:
            #                         response_snts = sent_tokenize(response)
            #                         new_snts = []
            #                         for snt in response_snts:
            #                             f_snt = snt.replace(str(y), '')
            #                             if year_identifier(f_snt)!=None:
            #                                 print('removed year --> ', y)
            #                                 new_snts.append(f_snt)
            #                         if len(new_snts)>0:
            #                             response = ' '.join(new_snts)
            #                             return response
            #                         else:
            #                             return None
            #                     else:
            #                         return response

            #                 # repeat
            #                 q_years = ex['years'] # question dates
            #                 time_relation = ex['time_relation'].lower()
            #                 implicit_condition = ex['implicit_condition']
            #                 if time_relation in ['before','as of','by','until']:
            #                     time_relation_type = 'before'
            #                 elif time_relation == 'from':
            #                     if len(q_years)==1:
            #                         time_relation_type = 'after'
            #                     else:
            #                         time_relation_type = 'between'
            #                 elif time_relation == 'since':
            #                     time_relation_type = 'after'
            #                 elif time_relation in ['after','between']:
            #                     time_relation_type = time_relation
            #                 else:
            #                     time_relation_type = 'other'

            #                 # for y in summary_years:
            #                 #     if time_relation_type in ['before', 'other']:
            #                 #         if y > q_years[0]:
            #                 #             response = snt_temporal_filter(response, summary_years, y)
            #                 #     # Republican Jim Justice was elected governor in 2016 -->  Who heads the Executive Department of the West Virginia government after April 5, 2018? 
            #                 #     # elif time_relation_type == 'after':
            #                 #     #     if y < q_years[0] and all([w not in response for w in ['since', 'from']]):
            #                 #     #         response = snt_temporal_filter(response, summary_years, y)
            #                 #     elif time_relation_type == 'between':
            #                 #         if y < min(q_years) or y > max(q_years):
            #                 #             response = snt_temporal_filter(response, summary_years, y)
            #                 # print('AFTER: ', response)

            #             else:
            #                 # no years in the sentence
            #                 pass
            #             # import ipdb; ipdb.set_trace()

            #         if response and response not in new_texts:
            #             new_texts.append(response)



            snts = ex['top_snt_id']
            ctx_map = {ctx['id']:ctx for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]}
            # qfs_map = {ctx['id']:ctx['QFS_summary'] for ctx in ex['snt_hybrid_rank'][:args.ctx_topk] if ctx['QFS_summary']}
            new_texts = []
            ctx_ids = []
            for ctx_id, snt, _ in snts:
                if ctx_id not in ctx_map or len(ctx_ids)>=args.ctx_topk:
                    break
                if ctx_id not in ctx_ids:
                    ctx_ids.append(ctx_id)
                ctx = ctx_map[ctx_id]
                if ctx['QFS_summary'] not in ['', None]:
                    decontext_snts = [ctx['QFS_summary']]
                else:
                    snt = snt[len(ctx['title']):].strip()

                    prompt = decontext(ctx['title'], ctx['text'], snt)
                    responses = call_pipeline(args, [prompt], 200)
                    decontext_snts = responses[0]
                    print('##\n',prompt,'\n',decontext_snts)
                    import ipdb; ipdb.set_trace()
                    
                # repeat
                q_years = ex['years'] # question dates
                time_relation = ex['time_relation'].lower()
                implicit_condition = ex['implicit_condition']
                if time_relation in ['before','as of','by','until']:
                    time_relation_type = 'before'
                elif time_relation == 'from':
                    if len(q_years)==1:
                        time_relation_type = 'after'
                    else:
                        time_relation_type = 'between'
                elif time_relation == 'since':
                    time_relation_type = 'after'
                elif time_relation in ['after','between']:
                    time_relation_type = time_relation
                else:
                    time_relation_type = 'other'


                for snt in decontext_snts:
                    if len(snt.split())<3:
                        continue
                    snt_ = snt.lower()
                    snt_years = year_identifier(snt)
                    snt_time_relation_type = None
                    has_from = False
                    has_until = False
                    if snt_years:
                        for y in snt_years:
                            if f'in {y}' in snt_ or f'in the {y}' in snt_:
                                snt_time_relation_type = 'other'
                            elif f'from {y}' in snt_:
                                snt_time_relation_type = 'after'
                                has_from = True
                            elif f'until {y}' in snt_:
                                snt_time_relation_type = 'before'
                                has_until = True
                    if has_from and has_until:
                        snt_time_relation_type = 'between'

                    filter_out = False
                    if snt_time_relation_type == 'before':
                        if time_relation_type in ['after','other'] and q_years[0]>snt_years[0]:
                            filter_out = True
                        elif time_relation_type == 'between' and min(q_years)>snt_years[0]:
                            filter_out = True
                     
                    elif snt_time_relation_type == 'after':
                        if time_relation_type in ['before','other'] and q_years[0]<snt_years[0]:
                            filter_out = True
                        elif time_relation_type == 'between' and max(q_years)<snt_years[0]:
                            filter_out = True

                    elif snt_time_relation_type == 'between':
                        if time_relation_type == 'before' and q_years[0]<min(snt_years):
                            filter_out = True
                        elif time_relation_type == 'after' and q_years[0]>max(snt_years):
                            filter_out = True
                        elif time_relation_type == 'between':
                            if min(snt_years)>max(q_years) or max(snt_years)<min(q_years):
                                filter_out = True
                        elif time_relation_type == 'other':
                            if q_years[0]>max(q_years) or q_years[0]<min(q_years):
                                filter_out = True
                    
                    elif snt_time_relation_type == 'other':
                        # 1977, 1978 and 1981 World Series
                        for y in snt_years:
                            if time_relation_type == 'before' and q_years[0]<y:
                                snt = snt.replace(str(y),'')
                            elif time_relation_type == 'after' and q_years[0]>y:
                                snt = snt.replace(str(y),'')
                            elif time_relation_type == 'between':
                                if y>max(q_years) or y<min(q_years):
                                    snt = snt.replace(str(y),'')
                            elif time_relation_type == 'other':
                                if y != q_years[0]:
                                    snt = snt.replace(str(y),'')
                        if year_identifier(snt) == None:
                            filter_out = True

                    print('==\n', snt)
                    print(snt_years)
                    print('type ', snt_time_relation_type)
                    print('filter ', filter_out)
                    # import ipdb; ipdb.set_trace()
                    if (not filter_out) and (snt not in new_texts):
                        new_texts.append(snt)


                # for y in summary_years:
                #     if time_relation_type in ['before', 'other']:
                #         if y > q_years[0]:
                #             response = snt_temporal_filter(response, summary_years, y)
                #     # Republican Jim Justice was elected governor in 2016 -->  Who heads the Executive Department of the West Virginia government after April 5, 2018? 
                #     # elif time_relation_type == 'after':
                #     #     if y < q_years[0] and all([w not in response for w in ['since', 'from']]):
                #     #         response = snt_temporal_filter(response, summary_years, y)
                #     elif time_relation_type == 'between':
                #         if y < min(q_years) or y > max(q_years):
                #             response = snt_temporal_filter(response, summary_years, y)
                # print('AFTER: ', response)


            # new_texts = new_texts[:1]

            # new_texts[0] = call_pipeline(args, [prompt])[0]
            # new_texts = new_texts[::-1]
            # for x in ex['top_snt_id'][:10]:
            #     print(x, '\n')
            # print('~~~')
            # new_texts = [ctx['title'] + ' | ' + ctx['text'] for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]]
    
            print('\n------\n',question,'\n------\n') 

            for x in new_texts:
                print(x,'\n')
            
            prompt = c_prompt(question, '\n\n'.join(new_texts))
            ans = call_pipeline(args, [prompt])


            rag_pred = ans[0]
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