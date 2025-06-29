import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse 
from temp_eval import normalize



# def GradeHallucinations(generation, document):
#     prompt = f"""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
# Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

# Set of facts: \n\n {document} \n\n LLM generation: {generation}
# """
#     return prompt

# def GradeAnswers(answer)

# """You are a grader assessing whether an answer addresses / resolves a question.
# Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""


def GradeDocuments(context, question):
    prompt = f"""You will be given a context paragraph and a question. Your task is decide whether the context is relevant and contains the answer to the question.
First read the paragraph after <Context> and question after <Question> carefully.
Then you should think step by step and give your thought after <Thought>.
Finally, write the response by "Yes" or "No" after <Response>.

There are some examples for you to refer to:
<Context>
1882 in rail transport | September 19 – Oliver Bulleid, chief mechanical engineer of the Southern Railway (Great Britain) 1937–1948, born in New Zealand (d. 1970). 
</Context>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Thought>
The mentions that Oliver Bulleid was the chief mechanical engineer of the Southern Railway (Great Britain) from 1937 to 1948, which answers the question about whom he was employed by. So the context is relevant to the question. Therefore, the response is "Yes".
</Thought>
<Response>
Yes
</Response>

<Context>
Petronas Towers | The Petronas Towers (Malay: Menara Berkembar Petronas), also known as the Petronas Twin Towers and colloquially the KLCC Twin Towers, are an interlinked pair of 88-storey supertall skyscrapers in Kuala Lumpur, Malaysia, standing at 451.9 metres (1,483 feet).
</Context>
<Question>
Tallest building in the world?
</Question>
<Thought>
The context about the Petronas Towers is not directly relevant to the question about the tallest building in the world. So the context is not relevant to the question. Therefore, the response is "No".
</Thought>
<Response>
No
</Response>

<Context>
List of 20th-century religious leaders Church of England | Formal leadership: Supreme Governor of the Church of England (complete list) – ; Victoria, Supreme Governor (1837–1901) ; Edward VII, Supreme Governor (1901–1910) ; George V, Supreme Governor (1910–1936) ; Cosmo Gordon Lang, Archbishop of Canterbury (1928–1942) ; William Temple, Archbishop of Canterbury (1942–1944) ; 
</Context>
<Question>
Who is the head of the Church in England?
</Question>
<Thought>
The context provides a list of historical figures who held the title of Supreme Governor, including monarchs such as Queen Victoria, King Edward VII, and King George V. So the context is relevant to the question. Therefore, the response is "Yes".
</Thought>
<Response>
Yes
</Response>

<Context>
Abbey Christian Brothers' Grammar School | Frank Aiken (1898-1983) TD, Irish Republican Army commander, Tánaiste, Minister for the Co-ordination of Defensive Measures (1939–45), Minister for Finance (1945–48) and Minister for External Affairs (1951–54; 1957–69) ; Séamus Mallon (1936-2020), Member of Parliament (MP) for Newry & Armagh (1986-2005)
</Context>
<Question>
Who is the Minister for Defence in Ireland?
</Question>
<Thought>
The provided context, which includes information about Frank Aiken and Séamus Mallon, is not directly relevant to the question about the Minister for Defence in Ireland. So the context is not relevant to the question. Therefore, the response is "No".
</Thought>
<Response>
No
</Response>

Now your context paragraph and question are
<Context>
{context}
</Context>
<Question>
{question}?
</Question>
<Thought>
"""
    return prompt


def CombinedReader(generations, question, short=False):

    prompt=f"""As an assistant, your task is to answer the question based on the given knowledge. Answer the given question, you can refer to the document provided. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tag. You can refer to the knowledge to answer the question.
Answer only the name for 'Who' questions.
If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:
<Context>
The Newton D. Baker House in Washington, D.C. was owned by the following individuals over time: Thomas Beall from 1794 to 1796, John Laird from 1796, George Peter to 1827, and David W. Hudgens from 2017.
</Context>
<Question>
Who owned the Newton D. Baker House in Washington, D.C on 2 Jan 1795? 
</Question>
<Thought>
According to the context, the Newton D. Baker House in Washington, D.C. was owned by Thomas Beall from 1794 to 1796. 2 Jan 1795 is between 1794 and 1796. Therefore, the answer is Thomas Beall. 
</Thought>
<Answer>
Thomas Beall
</Answer>

<Context>
In 1977, Trump married Czech model Ivana Zelníčková. The couple divorced in 1990, following his affair with actress Marla Maples.

Trump and Maples married in 1993 and divorced in 1999.

In 2005, Donald Trump married Slovenian model Melania Knauss. They have one son, Barron (born 2006).
</Context>
<Question>
Who was the spouse of Donald Trump between 2010 and 2014?
</Question>
<Thought>
According to the context, Donald Trump married Melania Knauss in 2005. The period between 2010 and 2014 is after 2005. Therefore, the answer is Melania Knauss. 
</Thought>
<Answer>
Melania Knauss
</Answer>

<Context>
Dwight Howard played for Orlando Magic from 2004 to 2012.

Dwight Howard played for Los Angeles Lakers from 2012.

On November 21, 2020, the Philadelphia 76ers signed Howard to a one-year deal.
</Context>
<Question>
Dwight Howard played for which team in 2012?
</Question>
<Thought>
According to the context, Dwight Howard transferred from Orlando Magic to Los Angeles Lakers in 2012. Therefore, the answer is Los Angeles Lakers. 
</Thought>
<Answer>
Los Angeles Lakers
</Answer>

<Context>
Theo-Ben Gurirab served as the second Prime Minister of Namibia from 28 August 2002 to 20 March 2005, following the demotion and subsequent resignation of Hage Geingob.

Theo-Ben Gurirab was Associate Representative of the SWAPO Mission to the United Nations and the United States from 1964 to 1972

Saara Kuugongelwa-Amadhila (born 12 October 1967) is a Namibian politician who has served as the Prime Minister of Namibia since 2015.
</Context>
<Question>
Theo-Ben Gurirab took which position as of 2004?
</Question>
<Thought>
According to the context, Theo-Ben Gurirab served as the Prime Minister of Namibia from 28 August 2002 to 20 March 2005. 2004 is between 28 August 2002 and 20 March 2005. Therefore, the answer is Prime Minister of Namibia.
</Thought>
<Answer>
Prime Minister of Namibia
</Answer>

<Context>
England national football team has qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>
According to the context, England got to the semi final of a World Cup in 1990 and 2018. 2018 is the last time before 2019. Therefore, the answer is 2018. 
</Thought>
<Answer>
Prime Minister of Namibia
</Answer>

<Context>
Super Bowl XL was an American football game between Seattle Seahawks and Pittsburgh Steelers to decide the National Football League (NFL) champion for the 2005 season. The Steelers defeated the Seahawks by the score of 21–10. The game was played on February 5, 2006.

The winner of Super Bowl XLIII, which took place on February 1, 2009, was the Pittsburgh Steelers.

Pittsburgh Steelers won the Super Bowl XLIII for the 2008 NFL season.
</Context>
<Question>
When was the first season that Pittsburgh Steelers won Super Bowl championship between 2000 and 2010?
</Question>
<Thought>
According to the context, Pittsburgh Steelers won Super Bowl championship for 2005 and 2008 seasons. 2005 is the first time between 2000 and 2010. Therefore, the answer is 2005. 
</Thought>
<Answer>
2005
</Answer>"""
    
    extend = """
<Context>
A. P. J. Abdul Kalam served as the President of India from 2002 to 2007.

K. R. Narayanan served as the President of India from 1997 until 2002.

Droupadi Murmu served as the 15th President of India from 2021.
</Context>
<Question>
Who was the first President of India between 2006 and 2022?
</Question>
<Thought>
According to the context, between 2006 and 2022, A. P. J. Abdul Kalam was the President of India from 2006 to 2007 and Droupadi Murmu was the President of India from 2021 to 2022. A. P. J. Abdul Kalam is the first President of India between 2006 and 2022. Therefore, the answer is A. P. J. Abdul Kalam. 
</Thought>
<Answer>
A. P. J. Abdul Kalam
</Answer>

<Context>
Sam Nujoma: president of Namibia (1990-2005)

Hifikepunye Pohamba was the president of Namibia between 2005 and 2015.

Hage Geingob is the incumbent president of Namibia from 2015.
</Context>
<Question>
Who was the last Namibia's president from 29 January 2002 to January 2016?
</Question>
<Thought>
According to the context, from 29 January 2002 to January 2016, Sam Nujoma was the Namibia's president from 29 January 2002 to 2005, Hifikepunye Pohamba was the Namibia's president from 2005 to 2015, and Hage Geingob was the Namibia's president from 2015 to January 2016. Hage Geingob was the last Namibia's president from 29 January 2002 to January 2016. Therefore, the answer is Hage Geingob.
</Thought>
<Answer>
Hage Geingob
</Answer>

<Context>
Hifikepunye Pohamba was the president of Namibia between 2005 and 2016 and the chief of the Army from 2012 to 2014.
</Context>
<Question>
What is the last position Hifikepunye Pohamba took between 2012 and 2015?
</Question>
<Thought>
According to the context, Hifikepunye Pohamba was the president of Namibia and the chief of the Army from 2012 to 2014. Hifikepunye Pohamba was the president of Namibia from 2014 to 2015. The last position Hifikepunye Pohamba took was the president of Namibia. Therefore, the answer is president of Namibia.
</Thought>
<Answer>
president of Namibia
</Answer>"""

    ask = f"""

Now your question and context knowledge are
<Context>
{generations}
</Context>
<Question>
{question}
</Question>
<Thought>
"""
    if not short:
        prompt += extend
    prompt += ask
    return prompt



def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_metriever_bgegemma_llama_8b_qfs5_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="situatedqa_contriever_bgegemma_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=3)
    parser.add_argument('--param-pred', type=bool, default=True)
    parser.add_argument('--param-cot', type=bool, default=False)
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--save-note', type=str, default=None)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--reader', type=str, default='gpt', choices=['llama', 'timo', 'timellama','llama_70b','llama_8b', 'gpt'])
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

    if 'gpt' in args.l.lower():
        pass
    else:
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=4, quantization="AWQ", max_model_len=20000)
        else:
            mx_len = 2048 if args.reader=='timo' else 20000
            args.llm = LLM(args.l, tensor_parallel_size=2, max_model_len=mx_len)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    # examples = examples[:100]

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]

    # x = "Who was the last to win the Kentucky Derby in under 2 minutes between 1973 and 2010?"
    # examples = [ex for ex in examples if x in ex['question']]

    examples = [ex for ex in examples if ex['time_relation'] != '']
    if len(examples)==0:
        print(f'\n!! find no example in top {args.max_examples}.')
    
    ########  QA  ######## 
    if args.param_pred:
        if args.param_cot:
            prompts = [zc_cot_prompt(ex['question']) for ex in examples]
        else:
            prompts = [zc_prompt(ex['question']) for ex in examples]
        param_preds = call_pipeline(args, prompts, 400)
        print('zero context prediction finished.')

    tmp_key = args.ctx_key_s2 if args.ctx_key_s2 else args.ctx_key_s1

    if 'gpt' in args.l.lower():
        for k, ex in enumerate(examples):
            ex['rag_pred'] = ''
            ex['rag_acc'] = 0
            ex['rag_f1'] = 0
    else:
        if args.paradigm=='concat':

            prompts, texts = [], []
            for ex in examples:
                text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
                texts.append(text)
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
                    checker_prompt = GradeDocuments(context, normalized_question)
                    checker_prompts.append(checker_prompt)

            checker_responses = call_pipeline(args, checker_prompts, 400)
            checker_results = ['yes' in res.lower() for res in checker_responses]
            print('\nstarted reader.\n')

            if args.reader=='timo':
                prompts, texts = [], []
                for ex in examples:
                    ctx_list=[]
                    question = ex['question']
                    normalized_question = ex['normalized_question']
                    for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                        checker_result = checker_results.pop(0)
                        if checker_result:
                            ctx_list.append(ctx)

                    text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ctx_list[:min(3,len(ctx_list))]])
                    texts.append(text)
                    prompt = c_prompt(ex['question'], text)
                    prompts.append(prompt)

                rag_preds = call_pipeline(args, prompts, 500)
                print(f'{tmp_key} top {args.ctx_topk} contexts prediction finished.')
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
                generation_prompts = []
                questions = []
                for ex in examples:
                    question = ex['question']
                    normalized_question = ex['normalized_question']
                    for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                        checker_result = checker_results.pop(0)
                        if checker_result:
                            doc = ctx['title'] + ' | ' + ctx['text'].strip()
                            generation_prompt = LLMGenerations(doc, normalized_question, args.reader=='timo')
                            generation_prompts.append(generation_prompt)
                            questions.append(question)
                generation_responses = call_pipeline(args, generation_prompts, 400, ver=True)
                buf = {}
                for question, gen in zip(questions, generation_responses):
                    if question not in buf:
                        buf[question] = []
                    if gen not in buf[question]:
                        if len(gen.split())<50:
                            buf[question].append(gen)

                for question in buf:
                    summarizations = buf[question]
                    years = [year_identifier(s) for s in summarizations]
                    filtered_data = [(s, y) for s, y in zip(summarizations, years) if y is not None]
                    filtered_data = [(s, min(y)) for s,y in filtered_data]
                    sorted_data = sorted(filtered_data, key=lambda pair: pair[1])
                    sorted_summarizations, sorted_years = zip(*sorted_data) if sorted_data else ([], [])
                    buf[question] = sorted_summarizations

                combined_prompts = []
                for ex in examples:
                    question = ex['question']
                    normalized_question = ex['normalized_question']
                    if question not in buf:
                        buf[question] = []
                    combined_prompt = CombinedReader('\n\n'.join(buf[question]), question, args.reader=='timo')
                    combined_prompts.append(combined_prompt)
                combined_responses = call_pipeline(args, combined_prompts, 400)
                for k, ex in enumerate(examples):
                    question = ex['question']
                    if len(buf[question])==0:
                        combined_responses[k]=''

                for k, ex in enumerate(examples):
                    question = ex['question']
                    gold_evidences = ex['gold_evidences']
                    rag_pred = combined_responses[k]
                    rag_pred = force_string(rag_pred)
                    print(f'\n----{k}-----')
                    if check_no_knowledge(rag_pred):
                        print('parallel read no knowledge.')
                        text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in ex[tmp_key][:args.ctx_topk]])
                        prompt = c_prompt(ex['question'], text)
                        rag_pred = call_pipeline(args, [prompt], 500)[0]
                        rag_pred = force_string(rag_pred)
                        if len(rag_pred.split())>50:
                            rag_pred=''

                    if check_no_knowledge(rag_pred):
                        print('concat read no knowledge.')
                        prompt = zc_cot_prompt(question)
                        param_pred = call_pipeline(args, [prompt], 400)[0]
                        rag_pred = force_string(param_pred)
                        if len(rag_pred.split())>50:
                            rag_pred=''

                    ex['rag_pred'] = rag_pred
                    ex['rag_acc'] = int(normalize(rag_pred) in [normalize(ans) for ans in ex['answers']])
                    ex['rag_f1'] = max_token_f1([normalize(ans) for ans in ex['answers']], normalize(rag_pred))
                    print(question)
                    print(combined_prompts[k].split('Now your question and context knowledge are')[-1])
                    print('\n',rag_pred, ex['rag_acc'])


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