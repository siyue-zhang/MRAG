import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import ray
# ray.init(num_gpus=4) 

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


def LLMGenerations(document, qeustion):
    prompt = f"""You are a summarizer summarizing a retrieved document about a user question. Keep the key dates in the summarization. Write "None" if the document has no relevant content about the question.

There are some examples for you to refer to:
<Document>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Document>
<Question>
When did the Houston Rockets win the NBA championship?
</Question>
<Summarization>
The Houston Rockets won the NBA championship twice, in 1994 and 1995.
</Summarization>

<Document>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Document>
<Question>
Who serve as President of India?
</Question>
<Summarization>
Neelam Sanjiva Reddy became the sixth President in 1977. K. R. Narayanan, the first Dalit president, served from 1997 to 2002. In 2022, Droupadi Murmu became the 15th President and the first tribal woman to hold the position.
</Summarization>

<Document>
The Lost World: Jurassic Park | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America.
</Document>
<Question>
What was the worldwide box office of Jurassic movie?
</Question>
<Summarization>
The worldwide box office for The Lost World: Jurassic Park (1997) was $618.6 million.
</Summarization>

<Document>
2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
</Document>
<Question>
Who won the Grand National?
</Question>
<Summarization>
None
</Summarization>

<Document>
Oliver Bulleid |  He was born in Invercargill, New Zealand, to William Bulleid and his wife Marian Pugh, both British immigrants. On the death of his father in 1889, his mother returned to Llanfyllin, Wales, where the family home had been, with Bulleid. In 1901, after a technical education at Accrington Grammar School, he joined the Great Northern Railway (GNR) at Doncaster at the age of 18, as an apprentice under H. A. Ivatt, the Chief Mechanical Engineer (CME). After a four-year apprenticeship, he became the assistant to the Locomotive Running Superintendent, and a year later, the Doncaster Works manager. In 1908, he left to work in Paris with the French division of Westinghouse Electric Corporation as a Test Engineer, and was soon promoted to Assistant Works Manager and 
</Document>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Summarization>
Oliver Bulleid was an employee of the Great Northern Railway (GNR) from 1901 and the Westinghouse Electric Corporation from 1908.
</Summarization>

<Document>
Doris Schröder-Köpf | Köpf and partner Sven Kuntze moved to New York City in 1990, where they had a daughter named Klara in the following year. Soon after the birth the pair separated and Köpf moved back to Bavaria with the child. In October 1997, Köpf married Gerhard Schröder, then Minister-President of Lower Saxony.
</Document>
<Question>
Who was the spouse of Doris Schröder?
</Question>
<Summarization>
Doris Schröder-Köpf married Gerhard Schröder, then Minister-President of Lower Saxony, in October 1997.
</Summarization>

<Document>
Newton D. Baker House | 1794-1796 - Thomas Beall ; 1796-? - John Laird  ; ?-1827 - George Peter ; 2017-present - David W. Hudgens
</Document>
<Question>
Who owned the Newton D. Baker House in Washington DC?
</Question>
<Summarization>
The Newton D. Baker House in Washington, D.C. was owned by the following individuals over time: Thomas Beall from 1794 to 1796, John Laird from 1796, George Peter to 1827, and David W. Hudgens from 2017.
</Summarization>

<Document>
Intel | Intel embarked on a 10-year period of unprecedented growth as the primary and most profitable hardware supplier to the PC industry, part of the winning 'Wintel' combination. Moore handed over his position as CEO to Andy Grove in 1987. By launching its Intel Inside marketing campaign in 1991, Intel was able to associate brand loyalty with consumer selection, so that by the end
</Document>
<Question>
Who was the CEO of Intel?
</Question>
<Summarization>
Moore was the CEO of Intel before 1987 and Andy Grove was the CEO of Intel after 1987.
</Summarization>

Now your document and question are
<Document>
{document}
</Document>
<Question>
{qeustion}
</Question>
<Summarization>
"""
    return prompt


def GradeDocuments(context, question):
    prompt = f"""You will be given a context paragraph and a question. Your task is decide whether the context is relevant and contains the answer to the question.
Requirements are follows:
- First read the paragraph after <Context> and question after <Question> carefully.
- Then you should think step by step and give your thought after <Thought>.
- Finally, write the response by "Yes" or "No" after <Response>.

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
Abbey Christian Brothers' Grammar School | Frank Aiken (1898-1983) TD, Irish Republican Army commander, Tánaiste and served as Minister for Defence (1932–39), Minister for the Co-ordination of Defensive Measures (1939–45), Minister for Finance (1945–48) and Minister for External Affairs (1951–54; 1957–69) ; Séamus Mallon (1936-2020), Member of Parliament (MP) for Newry & Armagh (1986-2005)
</Context>
<Question>
Who is the Minister for Defence in Ireland?
</Question>
<Thought>
The provided context, which includes information about Frank Aiken and Séamus Mallon, is not directly relevant to the question about the current Minister for Defence in Ireland. So the context is not relevant to the question. Therefore, the response is "No".
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


def CombinedReader(generations, question):

    prompt=f"""As an assistant, your task is to answer the question based on the given knowledge. Answer the given question, you can refer to the document provided. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
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
According to the context, Donald Trump married Slovenian model Melania Knauss in 2005. The period between 2010 and 2014 is after 2005. Therefore, the answer is Melania Knauss. 
</Thought>
<Answer>
Melania Knauss
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
</Answer>

<Context>
A. P. J. Abdul Kalam served as the President of India from 2002 to 2007.

K. R. Narayanan served as the President of India from 1997 until 2002.

Droupadi Murmu served as the 15th President of India from 2022.
</Context>
<Question>
Who was the first President of India between 2006 and 2022?
</Question>
<Thought>
According to the context, A. P. J. Abdul Kalam was the President of India in 2006. A. P. J. Abdul Kalam is the first President of India between 2006 and 2022. Therefore, the answer is A. P. J. Abdul Kalam. 
</Thought>
<Answer>
A. P. J. Abdul Kalam
</Answer>

Now your question and context knowledge are
<Context>
{generations}
</Context>
<Question>
{question}
</Question>
<Thought>
"""
    return prompt



def main():
    parser = argparse.ArgumentParser(description="Reader")
    parser.add_argument('--max-examples', type=int, default=10)
    parser.add_argument('--retriever-output', type=str, default="timeqa_contriever_metriever_bgegemma_llama_8b_qfs5_outputs.json")
    # parser.add_argument('--retriever-output', type=str, default="timeqa_contriever_minilm12_outputs.json")
    parser.add_argument('--ctx-topk', type=int, default=5)
    parser.add_argument('--param-pred', type=bool, default=False)
    parser.add_argument('--param-cot', type=bool, default=False)
    parser.add_argument('--not-save', type=bool, default=True)
    parser.add_argument('--save-note', type=str, default=None)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
    )
    parser.add_argument('--reader', type=str, default='llama_8b', choices=['llama', 'timo', 'timellama','llama_70b','llama_8b'])
    parser.add_argument('--paradigm', type=str, default='fusion', choices=['fusion', 'concat'])

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
        args.llm = LLM(args.l, tensor_parallel_size=1, max_model_len=mx_len)

    # load examples
    if 'retrieved' not in args.retriever_output:
        args.retriever_output = f'./retrieved/{args.retriever_output}'
    examples = load_json_file(args.retriever_output)
    print('examples loaded.')

    examples = examples[100:200]

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]

    # x = "What was the official name of Tinkoff (cycling team) in 2003?"
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
            prompt = c_cot_prompt(ex['question'], text)
            # prompt = c_prompt(ex['question'], text)
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

        generation_prompts = []
        questions = []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                checker_result = checker_results.pop(0)
                if checker_result:
                    doc = ctx['title'] + ' | ' + ctx['text'].strip()
                    generation_prompt = LLMGenerations(doc, normalized_question)
                    generation_prompts.append(generation_prompt)
                    questions.append(question)
        generation_responses = call_pipeline(args, generation_prompts, 400, ver=True)
        buf = {}
        for question, gen in zip(questions, generation_responses):
            if question not in buf:
                buf[question] = []
            if gen not in buf[question]:
                buf[question].append(gen)

        assert 1==2


        combined_prompts = []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']
            combined_prompt = CombinedReader('\n\n'.join(buf[question]), question)
            combined_prompts.append(combined_prompt)
        combined_responses = call_pipeline(args, combined_prompts, 400, ver=True)
            
        assert 1==2




        # for x, y in zip(checker_prompts, checker_responses):
        #     print('\n==\n')
        #     print(x.split('Now your context paragraph and question are')[-1])
        #     print(y)

        prompts, texts = [], []
        for ex in examples:
            question = ex['question']
            normalized_question = ex['normalized_question']
            rel = []
            for ctx in ex['snt_hybrid_rank'][:args.ctx_topk]:
                checker_result = checker_results.pop(0)
                if checker_result:
                    rel.append(ctx)

            text = '\n\n'.join([ctx['title'] + ' | ' + ctx['text'].strip() for ctx in rel])
            texts.append(text)
            prompt = c_prompt(ex['question'], text)
            prompts.append(prompt)

        rag_preds = call_pipeline(args, prompts, 500)

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
    
    print(rag_preds)

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