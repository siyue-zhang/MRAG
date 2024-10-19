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

def call_pipeline(args, prompts):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
    outputs = args.llm.generate(prompts, sampling_params)
    # print('~~~')
    # print(prompts[0],'\n<>')
    # print(outputs[0].outputs[0].text)
    # print('~~~')
    responses = [output.outputs[0].text for output in outputs]
    responses = [res.split('<doc>')[0] if '<doc>' in res else res for res in responses]
    responses = [res.split('</doc>')[0] if '</doc>' in res else res for res in responses]
    responses = [res.split('Note:')[0] if 'Note:' in res else res for res in responses]
    responses = [res.split('Question:')[0] if 'Question:' in res else res for res in responses]
    responses = [res.split('<Question>')[0] if '<Question>' in res else res for res in responses]
    responses = [res.split('</Question>')[0] if '</Question>' in res else res for res in responses]
    responses = [res.split('<Summarization>')[0] if '<Summarization>' in res else res for res in responses]
    responses = [res.split('\n')[0].strip() for res in responses]
    # responses = [res.replace('\n','').strip() for res in responses]
    return responses 

def c_bracket_prompt(query, texts):

    prompt=f"""Answer the given question, you can refer to the document provided, especially focusing and analyzing on the sentence between brackets "[[text]]".
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer>.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question.
If the knowledge does not contain the answer, answer the question directly.
There are some examples for you to refer to:
<doc>
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | [[The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.]]

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</doc>
<Question>: When did England last get to the semi final of a World Cup before 2019?
<Answer>:
2018
<doc>
Bowl LV | [[For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan.]] They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</doc>
<Question>: Who sang the national anthem in the last Super Bowl as of 2021?
<Answer>:
Eric Church and Jazmine Sullivan
<doc>
Rugby World Cup | Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

Rugby union | Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
</doc>
<Question>: Where was the last Rugby World Cup held between 2007 and 2016?
<Answer>:
England

Now your question and reference knowledge are as follows.
<doc>
{texts}
</doc>
<Question>: {query}
<Answer>:
"""
    return prompt


# query focused summarizer
def get_QFS_prompt(question, title, text):
    prompt = f"""You are given a paragraph and a specific question. Your goal is to summarize the paragraph (between <doc> and </doc>) in complete sentences by answering the given question. If dates are mentioned in the paragraph, include them in your answer. If the question cannot be answered based on the paragraph, respond with "None." Ensure that the response is concise and directly addresses the question.
There are some examples for you to refer to:
<doc>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</doc>
<Question>:
When did the Houston Rockets win the NBA championship
<Summarization>:
The Houston Rockets won the NBA championship in 1994 and 1995.
<doc>
2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
</doc>
<Question>:
Who won the Grand National
<Summarization>:
None
<doc>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</doc>
<Question>:
Who serve as President of India
<Summarization>:
Neelam Sanjiva Reddy served as President in 1977, K. R. Narayanan in 1997, and Droupadi Murmu in 2022.

Now your question and paragraph are as follows.
<doc>
{title} | {text}
</doc>
<Question>:
{question}
<Summarization>:
"""
    return prompt

def c_prompt(query, texts):

    prompt=f"""Answer the given question, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer>.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question.
If the knowledge does not contain the answer, answer the question directly.
There are some examples for you to refer to:
<doc>
hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</doc>
<Question>: When did England last get to the semi final of a World Cup before 2019?
<Answer>:
2018
<doc>
For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</doc>
<Question>: Who sang the national anthem in the last Super Bowl as of 2021?
<Answer>:
Eric Church and Jazmine Sullivan
<doc>
Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
</doc>
<Question>: Where was the last Rugby World Cup held between 2007 and 2016?
<Answer>:
England

Now your question and reference knowledge are as follows.
<doc>
{texts}
</doc>
<Question>: {query}
<Answer>:
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json")
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
    parser.add_argument('--reader', type=str, default='timellama', choices=['rag', 'metriever', 'timo', 'timellama', 'extract_code'], help="Choose a reader option")

    args = parser.parse_args()
    args.l = llm_names(args.llm, instruct=False)
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
        examples = examples[:min(len(examples),args.max_examples)]

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
        if ex['time_relation'] != '':
            new_texts = []
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                normalized_question = ex['normalized_question']
                prompt = get_QFS_prompt(normalized_question, ctx['title'], ctx['text'])

                responses = call_pipeline(args, [prompt])
                response = responses[0].replace('\n','').strip()
                if 'none' not in response.lower():
                    # new_texts.append(ctx['title']+' | '+response)
                    new_texts.append(response)

#             snts = ex['top_snt_id']
#             ctx_map = {ctx['id']:ctx for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]}
#             qfs_map = {ctx['id']:ctx['QFS_summary'] for ctx in ex['snt_hybrid_rank'][:args.ctx_topk] if ctx['QFS_summary']}
#             new_texts = []
#             for ctx_id, snt, _ in snts:
#                 if ctx_id not in ctx_map:
#                     break
#                 ctx = ctx_map[ctx_id]
#                 title = ctx['title']
#                 snt = title.join(snt.split(title)[1:]).strip()
#                 if ctx_id in qfs_map:
#                     response = qfs_map[ctx_id]
#                 else:
#                     text = title+' | '+ctx['text']
#                     prompt=f'''You are given a sentence extracted from a larger context. Your task is to rewrite the sentence so that it stands alone, containing all the necessary details (such as names, dates, and places) that would have been understood in the original paragraph. Make sure the new sentence is clear, complete, and unambiguous, providing the reader with all essential information even if they do not have access to the original paragraph.
# There are some examples for you to refer to:
# <doc>
# 1997 tech conference | At the tech conference in San Francisco, industry leaders gathered to discuss the future of electric vehicles and sustainable technology. Among the most anticipated speakers was Jane Smith, the CEO of EV Innovators Inc. She took the stage and shared exciting news about the company's plans. She announced the new product line, which includes a series of long-range electric vehicles aimed at both personal and commercial use. This announcement is expected to significantly impact the market in the coming year.
# </doc>
# Original Sentence:
# She announced the new product line, which includes a series of long-range electric vehicles aimed at both personal and commercial use.
# Standalone Sentence:
# Jane Smith, the CEO of EV Innovators Inc., announced the new product line of long-range electric vehicles aimed at both personal and commercial use at the tech conference in San Francisco in 1997.
# <doc>
# Los Angeles Lakers | The Los Angeles Lakers had a remarkable game on 1 Jan 2018, led by their star player LeBron James. Playing against the Golden State Warriors at the Staples Center, the Lakers dominated from the start. In the third quarter, LeBron hit a crucial three-pointer, putting his team ahead by 12 points. The crowd erupted as the team secured an important win in their push for the playoffs.
# </doc>
# Original Sentence:
# The crowd erupted as the team secured an important win in their push for the playoffs.
# Standalone Sentence:
# The crowd erupted as the Los Angeles Lakers secured an important win in their push for the playoffs in a game on on 1 Jan 2018.
# Now your task is
# <doc>
# {text}
# </doc>
# Original Sentence:
# {snt}
# Standalone Sentence:
# '''
#                     responses = call_pipeline(args, [prompt])
#                     response = responses[0].replace('\n','').strip()
#                     # print('\n\n\n',prompt)
#                     # print('\n', response)
#                 new_texts.append(title+' | '+response)
            print('------')
            print(question,'\n')
            for x in new_texts:
                print(x,'\n')
            prompt = c_prompt(question, '\n\n'.join(new_texts))
            ans = call_pipeline(args, [prompt])

            import ipdb; ipdb.set_trace()

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