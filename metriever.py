import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import time
from utils import *
from prompts import *
# import ipdb; ipdb.set_trace()
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

import argparse
from copy import deepcopy
from vllm import LLM

from nltk.tokenize import sent_tokenize
from torch import Tensor
import torch
import torch.nn.functional as F

debug = None
debug_question = None

# debug = 'He is best known as the drummer for American hard rock band Guns'
# debug_question = 'Who is the drummer for Guns and Roses after 2006?'

# debug = """Ann Ward, Brittani Kline, Lisa D'Amato, Sophie Sumner, Laura James, Jourdan Miller, Keith Carlos, Nyle DiMarco, India Gants, and Kyla Coleman) crowned "America's Next"""
# debug_question = "Who won the latest America's Next Top Model as of 2021?"

def main():
    parser = argparse.ArgumentParser(description="Metriever")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever','hybrid'], 
        default='contriever', #
        help='Choose a model for stage 1 retrival'
    )
    parser.add_argument(
        '--stage2-model', 
        choices=['metriever','minilm6','minilm12','bge','tinybert','bgegemma','electra','nv','nv2','jina', 'sfr', None], 
        # default='nv2', #
        default='metriever', #
        help='Choose a model for stage 2 re-ranking'
    )
    parser.add_argument(
        '--metriever-model', 
        choices=['minilm6','minilm12','bge','tinybert','bgegemma','nv2'], 
        default='nv2',
        help='Choose a model for metriever stage2 re-ranking'
    )
    parser.add_argument('--contriever-output', type=str, default="./TempRAGEval/contriever_output/TempRAGEval.json")
    parser.add_argument('--bm25-output', type=str, default="./TempRAGEval/BM25_output/TempRAGEval.json")
    parser.add_argument('--ctx-topk', type=int, default=100)
    parser.add_argument('--QFS-topk', type=int, default=5)
    parser.add_argument('--snt-topk', type=int, default=200)
    parser.add_argument('--complete-ctx-text', type=bool, default=False)
    parser.add_argument('--hybrid-score', type=bool, default=True)
    parser.add_argument('--hybrid-base', type=float, default=0)
    parser.add_argument('--snt-with-title', type=bool, default=True)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--save-note', type=str, default=None)
    parser.add_argument('--subset', type=str, default='timeqa')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--load-keywords', type=bool, default=False)

    args = parser.parse_args()
    args.m1 = retrival_model_names(args.stage1_model)
    args.m2 = retrival_model_names(args.stage2_model) if args.stage2_model is not None else None
    args.m3 = retrival_model_names(args.metriever_model)
    args.l = llm_names(args.llm, instruct=True)
    args.llm_name = deepcopy(args.llm)
    args.reader = None

    # load llm
    if args.m2=='metriever':
        flg = '70b' in args.llm_name
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ", max_model_len=4096)
        else:
            args.llm = LLM(args.l, tensor_parallel_size=1, max_model_len=4096, device="cuda:0")
        
    # load semantic ranker for stage 2 / metriever
    if args.m2:
        name = args.m3 if args.m2 == 'metriever' else args.m2
        if 'gemma' in name:
            from FlagEmbedding import FlagLLMReranker
            args.model = FlagLLMReranker(name, use_fp16=True, device='cuda:2')
        elif 'sfr' in name.lower():
            from transformers import AutoTokenizer, AutoModel
            from torch.nn import DataParallel
            # load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto').eval()
            args.model = model
            args.tokenizer = tokenizer
        elif 'bge' in name:
            from FlagEmbedding import FlagReranker
            args.model = FlagReranker(name, use_fp16=True)
        elif 'nv' in name:
            from transformers import AutoModel
            from torch.nn import DataParallel
            embedding_model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16)
            for module_key, module in embedding_model._modules.items():
                embedding_model._modules[module_key] = DataParallel(module, device_ids=[1])
            args.model = embedding_model
        elif 'jina' in name:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                name,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.to('cuda')
            model.eval()
            args.model = model
        else:
            from sentence_transformers import CrossEncoder
            args.model = CrossEncoder(name)

    # load examples
    if args.stage1_model == 'contriever':
        ctx_key = 'ctxs'
        path = args.contriever_output
        examples = load_contriever_output(path)
    elif args.stage1_model == 'bm25':
        ctx_key = 'bm25_ctxs'
        path = args.bm25_output
        with open(path, 'r', encoding="utf-8") as file:
            examples = json.load(file)
    else:
        # hybrid
        ctx_key = 'hybrid_ctxs'
        path_contriever = args.contriever_output
        examples_contriever = load_contriever_output(path_contriever)

        path_bm25 = args.bm25_output
        with open(path_bm25, 'r') as file:
            examples_bm25 = json.load(file)

        assert len(examples_contriever) == len(examples_bm25)
        ctx_map = {}
        for idx in range(len(examples_contriever)):
            ctxs = examples_contriever[idx]['ctxs']
            ranked_chunk_ids = [ctx['id'] for ctx in ctxs[:min(1000, len(ctxs))]]
            for ctx in ctxs:
                if ctx['id'] not in ctx_map:
                    ctx_map[ctx['id']] = ctx
            
            bm25_ctxs = examples_bm25[idx]['bm25_ctxs']
            ranked_bm25_chunk_ids = [ctx['id'] for ctx in bm25_ctxs[:min(1000, len(bm25_ctxs))]]
            for ctx in bm25_ctxs:
                if ctx['id'] not in ctx_map:
                    ctx_map[ctx['id']] = ctx

            chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
            chunk_id_to_score = {}

            semantic_weight = 0.8
            bm25_weight = 0.2
            # Initial scoring with weights
            for chunk_id in chunk_ids:
                score = 0
                if chunk_id in ranked_chunk_ids:
                    index = ranked_chunk_ids.index(chunk_id)
                    score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
                if chunk_id in ranked_bm25_chunk_ids:
                    index = ranked_bm25_chunk_ids.index(chunk_id)
                    score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
                chunk_id_to_score[chunk_id] = score

            # Sort chunk IDs by their scores in descending order
            sorted_chunk_ids = sorted(
                chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
            )

            # Assign new scores based on the sorted order
            for index, chunk_id in enumerate(sorted_chunk_ids):
                chunk_id_to_score[chunk_id] = 1 / (index + 1)

            new_ctxs = [ctx_map[id] for id in sorted_chunk_ids]
            for ctx in new_ctxs:
                ctx['score'] = chunk_id_to_score[ctx['id']]
            del examples_contriever[idx]['ctxs']
            examples_contriever[idx][ctx_key] = new_ctxs
        examples = examples_contriever

    if 'norm' in args.contriever_output.lower():
        for ex in examples:
            ex['question'] = ex['ori_question']


    # examples = examples[300:365]
    # examples = [ex for ex in examples if 'he married Marjorie Ivatt?' not in ex['question']]

    # only keep situatedqa and timeqa samples for this code
    if args.subset == 'timeqa':
        examples = [ex for ex in examples if ex["source"] == 'timeqa']
        print('\nkeep only TimeQA subset.')
    elif args.subset == 'situatedqa':
        examples = [ex for ex in examples if ex["source"] == 'situatedqa']
        print('\nkeep only SituatedQA subset.')

    if debug_question:
        examples = [ex for ex in examples if ex['question']==debug_question]

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]

    # complete ctx head and tail sentences
    if args.complete_ctx_text:
        wiki_json = '/scratch/sz4651/Projects/metriever_final/enwiki-dec2021/psgs_w100.json'
        wiki = load_json_file(wiki_json)
        complete_ctx_map = {} # id to text
        for k, ex in enumerate(tqdm(examples, desc="Preprocessing contexts", total=len(examples))):
            ctxs = ex[ctx_key]
            complete_ctxs = []
            for ctx in ctxs:
                ctx_id = ctx['id']
                ctx_title = ctx['title']
                text = ctx['text'].strip()
                if ctx_id not in complete_ctx_map:
                    if ctx_title in wiki:
                        page = wiki[ctx_title]
                        flgs = [p['id'] == ctx_id for p in page]
                        page_has_ctx = any(flgs)==True
                        index_in_page = flgs.index(True) if page_has_ctx else None
                        if  page_has_ctx and index_in_page>0:
                            prev = page[index_in_page-1]
                            prev_text = prev['text'].strip()
                            if prev_text[-1] not in '.!?)}>':
                                ctx_sentences = sent_tokenize(prev_text)
                                ctx_sentences_clean = [s.strip() for s in ctx_sentences]
                                text = ctx_sentences_clean[-1] + ' ' + text
                        if page_has_ctx and index_in_page<(len(flgs)-1):
                            if text[-1] not in '.!?)}>':
                                after = page[index_in_page+1]
                                after_text = after['text'].strip()
                                ctx_sentences = sent_tokenize(after_text)
                                ctx_sentences_clean = [s.strip() for s in ctx_sentences]
                                text += ' ' + ctx_sentences_clean[0]
                    complete_ctx_map[ctx_id] = text
                ctx['text'] = complete_ctx_map[ctx_id]
                complete_ctxs.append(ctx)
            ex[ctx_key] = complete_ctxs

    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

    # from transformers import GPT2Tokenizer
    # # Load GPT-2 tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # words = []
    # for ex in examples_not_exact:
    #     tokenized_text = tokenizer.encode(ex['question'])
    #     token_length = len(tokenized_text)
    #     words.append(token_length)
    # print(np.mean(words))
    # import ipdb; ipdb.set_trace()

    #####################################################################################################################
    # Baselines 

    if args.m2 == None or args.m2 != 'metriever':
        print(f'--- Stage 1: {args.stage1_model} ---\n')
        print('\n**** Answers ****')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='answers')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='answers')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='answers')

        print(f'--- Stage 1: {args.stage1_model} ---\n')
        print('**** Gold Evidences ****')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        if args.m2 == None:
            return

    if args.m2 and args.m2 != 'metriever':
        # benchmark baselines

        flg = 'bge' in args.stage2_model or 'jina' in args.stage2_model
        start_time = time.time()
        for ex in tqdm(examples, desc="Reranking contexts"):
            question = ex['question']
            latest_ctxs = deepcopy(ex[ctx_key])
            latest_ctxs = latest_ctxs[:args.ctx_topk]

            if 'sfr' in args.m2.lower():
                task = 'Given a web search query, retrieve relevant passages that answer the query'
                queries = [get_detailed_instruct(task, question)]
                # No need to add instruction for retrieval documents
                passages = [ctx["title"]+" "+ctx["text"] for ctx in latest_ctxs]
                # get the embeddings
                max_length = 512
                input_texts = queries + passages
                batch_dict = args.tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = args.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                # Calculate similarity scores between the query and each passage
                query_embedding = embeddings[0]  # The first embedding is the query
                passage_embeddings = embeddings[1:]  # The rest are the passages
                # Compute similarity scores (cosine similarity)
                scores = (query_embedding @ passage_embeddings.T) 
                scores = scores.tolist()

            elif 'nv' not in args.m2:
                model_inputs = [[question, ctx["title"]+" "+ctx["text"]] for ctx in latest_ctxs]
                scores = args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)
            
            else:
                task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
                query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
                queries = [question]
                passage_prefix = ""
                passages = [ctx["title"]+" "+ctx["text"] for ctx in latest_ctxs]


                max_length = 512
                batch_size = 8

                # Encode and normalize the query
                query_embeddings = args.model.encode(
                    queries, instruction=query_prefix, max_length=max_length
                )
                query_embeddings = torch.tensor(query_embeddings)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

                # Encode passages in batches and compute similarity scores
                all_scores = []

                for i in range(0, len(passages), batch_size):
                    batch_passages = passages[i:i + batch_size]

                    passage_embeddings = args.model.encode(
                        batch_passages, instruction=passage_prefix, max_length=max_length
                    )
                    passage_embeddings = torch.tensor(passage_embeddings)
                    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

                    # Compute cosine similarity between query and current passage batch
                    scores = (query_embeddings @ passage_embeddings.T).view(-1)
                    all_scores.extend(scores.tolist())

                scores = all_scores

            for i, ctx in enumerate(latest_ctxs):
                ctx["score"] = float(scores[i])
            latest_ctxs = sorted(latest_ctxs, key=lambda x: x['score'], reverse=True)
            ex['reranker_ctxs'] = latest_ctxs
        
        end_time = time.time()
        duration = end_time - start_time
        duration /=len(examples)
        print(f"{args.m2} Baseline Execution Time: {duration:.6f} seconds")
        
        # evaluate reranking results
        ctx_key = 'reranker_ctxs'
        examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

        print('\n\n')
        print(f'--- Stage 2: {args.stage2_model} ---\n')
        print('\n**** Answers ****')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='answers')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='answers')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='answers')

        print(f'--- Stage 2: {args.stage2_model} ---\n')
        print('**** Gold Evidences ****')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        # save baseline results    
        save_json_file(f'./retrieved/{args.subset}_{args.stage1_model}_{args.stage2_model}_outputs.json', examples)
        return

    
    #####################################################################################################################
    # Metriever 

    # preprocess question about time
    start_time = time.time()
    for k, ex in enumerate(tqdm(examples, desc="Preprocessing time info", total=len(examples))):
        question = ex['question']
        ex['time_relation'] = ex['time_relation'].strip()
        time_relation = ex['time_relation'].lower()
        assert time_relation in question, question
        # if question != 'Who is the Speaker of the Karnataka Legislative Assembly from 31 July 2019?':
        #     continue

        time_relation_type = ''
        years, months = [], []
        no_time_question = question

        if time_relation != '':
            # classify time relation type
            # there are 4 types of time relation and 2 types of implicit condition (first and last)
            # 1)	before (before, as of, by until)
            # 2)	after (after, from, since)
            # 3)	between (between, from to)
            # 4)	other (in, on, around, during)

            # extract year and months
            date = ''
            parts = question.split(time_relation)
            no_time_question = time_relation.join(parts[:-1])
            date = parts[-1]
            # find year
            print('xx ', date)
            years = year_identifier(date)
            if len(years)>2:
                years=[min(years), max(years)]

            if len(years)>1:
                time_relation_type = 'between'
            elif time_relation in ['before','as of','by','until']:
                time_relation_type = 'before'
            elif time_relation in ['from','since','after']:
                time_relation_type = 'after'
            else:
                time_relation_type = 'other'

            # find months
            months = []
            def append_month(month_str):
                m = find_month(month_str)
                months.append(m if m else 0)
    
            if time_relation_type == 'between':
                delimiters = ['and', 'to', 'until']
                d_index = [d in date for d in delimiters]
                if any(d_index):
                    delimiter = delimiters[d_index.index(True)]
                    tmp = date.split(delimiter)
                    for w in tmp:
                        append_month(w.strip())
                else:
                    months = [0,0]
            else:
                append_month(date.strip())
            
        # 2 types of implicit condition (first and last)
        normalized_question, implicit_condition = remove_implicit_condition(no_time_question)
        normalized_question = normalized_question[:-1] if normalized_question[-1] in '.?!' else normalized_question

        print(f'\n==={k}===')
        print('Question : ', question)
        print('Normalized Question : ', normalized_question, '\n')
        
        ex['normalized_question'] = normalized_question
        ex['implicit_condition'] = implicit_condition
        ex['time_relation_type'] = time_relation_type
        ex['years'] = years # int
        ex['months'] = months

    if args.load_keywords:
        def load_example_keywords(path='./outputs/tmp_get_keywords.json'):
            with open(path, 'r') as file:
                return json.load(file)
        examples, question_keyword_map = load_example_keywords()
        print('loaded keywords.')
    else:
        # prepare keywords
        question_keyword_map = {}
        for ex in examples:
            normalized_question = ex['normalized_question']

            if normalized_question.startswith('How many times'):
                normalized_question = normalized_question.replace('How many times','When')
            elif normalized_question.startswith('How many'):
                normalized_question = normalized_question.replace('How many','What')
            ex['normalized_question'] = normalized_question

            question_keyword_map.setdefault(normalized_question, [])

        print('\nstart extracting keywords using llm.')
        prompts = [get_keyword_prompt(q) for q in question_keyword_map]
        # import ipdb; ipdb.set_trace()
        
        questions = [q for q in question_keyword_map]
        if args.llm_name not in ['gpt']:
            keyword_responses = call_pipeline(args, prompts)
        else:
            raise NotImplemented

        for i, q in enumerate(tqdm(questions, desc="Postprocessing keywords", total=len(questions))):
            tmp = keyword_responses[i][keyword_responses[i].index('['):(keyword_responses[i].index(']')+1)]
            tmp = eval(tmp)
            revised = []
            for kw in tmp:
                if kw in EXCL:
                    continue
                while kw.lower() not in q.lower():
                    # revise the extrcated keyword if not match with question
                    kw = ' '.join(kw.split()[:-1])
                if kw!='' and kw.lower() in q.lower():
                    revised.append(kw)
            revised = list(set(revised))
            revised = expand_keywords(revised, q, verbose=True)
            question_keyword_map[q] = revised
        
        save_json_file(path='./outputs/tmp_get_keywords.json', object=[examples, question_keyword_map])
        print('keywords saved.')


    if debug_question:
        examples = [ex for ex in examples if ex['question']==debug_question]

    end_time = time.time()
    duration = end_time - start_time
    duration /=len(examples)
    print(f"Preprocess Execution Time: {duration:.6f} seconds")
        
    # main reranking loop
    start_time = time.time()
    print('\nfinished preparation, start modular reranking.')
    all_QFS_prompts = []
    for k, ex in enumerate(examples):
        question = ex['question']
        time_relation = ex['time_relation']
        years = ex['years']
        implicit_condition = ex['implicit_condition']
        time_relation_type = ex['time_relation_type']
        normalized_question = ex['normalized_question']

        expanded_keyword_list, keyword_type_list = question_keyword_map[normalized_question]
        ex['expanded_keyword_list'] = expanded_keyword_list
        ex['keyword_type_list'] =  keyword_type_list
        print(f'\n---- {k} ----\n{question}\n')
        print(expanded_keyword_list)
        latest_ctxs = deepcopy(ex['ctxs']) # start from contriever top 1000

        if debug:
            for l,ctx in enumerate(ex['ctxs']):
                if debug in ctx['text']:
                    print(f'/////  contriever ctxs - {l}  /////')
                    print(ctx['text'])
                    break

        #####################################################################################################################
        # top 1000 ctx_keyword_rank_module
        ctx_kw_scores=[]
        for ctx in latest_ctxs:
            text = ctx['title'] + ' ' + ctx['text']
            ctx_score = count_keyword_scores(text, expanded_keyword_list, keyword_type_list)
            ctx_kw_scores.append((ctx, ctx_score))
        ctx_kw_scores = sorted(ctx_kw_scores, key=lambda x: x[1], reverse=True)
        latest_ctxs = [tp[0] for tp in ctx_kw_scores]

        if debug:
            for l,ctx in enumerate(latest_ctxs):
                if debug in ctx['text']:
                    print(f'/////  ctx_keyword_rank - {l}  /////')
                    print(ctx['text'])
                    break

        latest_ctxs = latest_ctxs[:args.ctx_topk] # only keep top 100
        ex['ctx_keyword_rank'] = latest_ctxs

        #####################################################################################################################
        # top 100 ctx_semantic_rank_module
        model_inputs = [[normalized_question, ctx["title"]+ ' ' + ctx["text"]] for ctx in latest_ctxs]
        if 'nv' in args.metriever_model:
            task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
            query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
            queries = [normalized_question]
            passage_prefix = ""
            passages = [x[1] for x in model_inputs]

            max_length = 512
            batch_size = 4

            # Encode and normalize the query
            query_embeddings = args.model.encode(
                queries, instruction=query_prefix, max_length=max_length
            )
            query_embeddings = torch.tensor(query_embeddings)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            # Encode passages in batches and compute similarity scores
            all_scores = []

            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i + batch_size]

                passage_embeddings = args.model.encode(
                    batch_passages, instruction=passage_prefix, max_length=max_length
                )
                passage_embeddings = torch.tensor(passage_embeddings)
                passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

                # Compute cosine similarity between query and current passage batch
                scores = (query_embeddings @ passage_embeddings.T).view(-1)
                all_scores.extend(scores.tolist())

            scores = all_scores
        else:
            flg = 'bge' in args.metriever_model
            scores = args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)  
        for i, ctx in enumerate(latest_ctxs):
            ctx["score"] = float(scores[i]) # update contriever score to reranker score
        latest_ctxs = sorted(latest_ctxs, key=lambda x: x["score"], reverse=True)
        ex['ctx_semantic_rank'] = latest_ctxs

        if debug:
            for l,ctx in enumerate(latest_ctxs):
                if debug in ctx['text']:
                    print(f'/////  ctx_semantic_rank - {l}  /////')
                    print(ctx['text'])
                    break

        #####################################################################################################################
        # top 200 snt_keyword_rank_module
        # add QFS summary for top semantic context
        
        # generate summaries
        QFS_prompts = []
        for ctx in latest_ctxs[:args.QFS_topk]:
            qfs_prompt = LLMGenerations(ctx['title']+' | '+ctx['text'],  normalized_question)
            # qfs_prompt = get_QFS_prompt(normalized_question, ctx['title'], ctx['text'])
            QFS_prompts.append(qfs_prompt)
        all_QFS_prompts += QFS_prompts

    all_summary_responses = call_pipeline(args, all_QFS_prompts, 200)

    end_time = time.time()
    duration = end_time - start_time
    duration /=len(examples)
    print(f"Retrieval Execution Time: {duration:.6f} seconds")
        
    start_time = time.time()
    for k, ex in enumerate(examples):

        question = ex['question']
        time_relation = ex['time_relation']
        years = ex['years']
        implicit_condition = ex['implicit_condition']
        time_relation_type = ex['time_relation_type']
        normalized_question = ex['normalized_question']

        summary_responses = all_summary_responses[k*args.QFS_topk:(k+1)*args.QFS_topk]
        latest_ctxs = ex['ctx_semantic_rank']
        expanded_keyword_list = ex['expanded_keyword_list']
        keyword_type_list = ex['keyword_type_list']

        get_ctx_by_id = {}
        sentence_tuples = []
        for idx, ctx in enumerate(latest_ctxs):
            get_ctx_by_id[ctx['id']] = ctx
            snts = sent_tokenize(ctx['text'])
            if args.snt_with_title:
                snts = [ctx['title']+' '+snt for snt in snts]
            if idx < args.QFS_topk:
                summary = summary_responses[idx]
                print('\n',question,' ',ex['answers'])
                print(ctx['text'])
                print(summary)
                if 'None' in summary:
                    summary = None
            else:
                summary = None
            if summary:
                snts.append(summary)
            ctx['QFS_summary'] = summary
            for snt in snts:
                snt = snt.strip()
                text = ctx['title'] + ' ' + snt
                snt_kw_score = count_keyword_scores(text, expanded_keyword_list, keyword_type_list)
                sentence_tuples.append((ctx['id'], snt, snt_kw_score))
        # sort all sentences including summaries
        sentence_tuples = sorted(sentence_tuples, key=lambda x: x[2], reverse=True)
        print(f'\ntotal {len(sentence_tuples)} sentences.')
        # get new ctx rank based on sentence rank 
        latest_ctxs = []
        id_included = []
        for ctx_id, snt, score in sentence_tuples:
            if ctx_id not in id_included:
                id_included.append(ctx_id)
                latest_ctxs.append(get_ctx_by_id[ctx_id])
        ex['snt_keyword_rank'] = latest_ctxs

        if debug:
            for l,ctx in enumerate(latest_ctxs):
                if debug in ctx['text']:
                    print(f'/////  snt_keyword_rank - {l}  /////')
                    print(ctx['text'])
                    break

        #####################################################################################################################
        # top 200 snt_hybrid_rank_module
        sentence_tuples_unchange = sentence_tuples[min(len(sentence_tuples),args.snt_topk):]
        sentence_tuples = sentence_tuples[:min(len(sentence_tuples),args.snt_topk)]
        print(f'rerank top {args.snt_topk} sentences.\n')

        if len(years)>0 and time_relation_type != 'other' and args.hybrid_score:
            # for hybrid ranking using question without time
            model_inputs = [[normalized_question, tp[1]] for tp in sentence_tuples]
        else:
            # rank by date matching
            model_inputs = [[question, tp[1]] for tp in sentence_tuples]

        if 'nv' in args.metriever_model:
            task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
            query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
            queries = [normalized_question]
            passage_prefix = ""
            passages = [x[1] for x in model_inputs]

            max_length = 512
            batch_size = 4

            # Encode and normalize the query
            query_embeddings = args.model.encode(
                queries, instruction=query_prefix, max_length=max_length
            )
            query_embeddings = torch.tensor(query_embeddings)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            # Encode passages in batches and compute similarity scores
            all_scores = []

            for i in range(0, len(passages), batch_size):
                batch_passages = passages[i:i + batch_size]

                passage_embeddings = args.model.encode(
                    batch_passages, instruction=passage_prefix, max_length=max_length
                )
                passage_embeddings = torch.tensor(passage_embeddings)
                passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

                # Compute cosine similarity between query and current passage batch
                scores = (query_embeddings @ passage_embeddings.T).view(-1)
                all_scores.extend(scores.tolist())

            semantic_scores = all_scores

        else:
            semantic_scores =  args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)
            semantic_scores = [float(s) for s in semantic_scores]

        if len(years)>0 and time_relation_type != 'other' and args.hybrid_score:
            # use temporal-semantic hybrid ranker
            # define spline for temporal coefficient
            spline = get_spline_function(time_relation_type, implicit_condition, years)
            # find closest year in the sentence and compute temporal coefficient based on closest year
            temporal_coeffs = get_temporal_coeffs(years, sentence_tuples, time_relation_type, implicit_condition, spline)
            final_scores = [args.hybrid_base*score + (1-args.hybrid_base)*score*coeff for score, coeff in zip(semantic_scores, temporal_coeffs)]
        else:
            # direct use semantic ranker
            final_scores = semantic_scores

        sentence_tuples = [(tp[0],tp[1],score) for score, tp in zip(final_scores, sentence_tuples)]
        sentence_tuples = sorted(sentence_tuples, key=lambda x: x[2], reverse=True)
        sentence_tuples += sentence_tuples_unchange

        ex['top_snt_id'] = sentence_tuples
        ex['top_snts'] = '\n\n'.join([tp[1] for tp in sentence_tuples[:20]])
        latest_ctxs = []
        id_included = []
        for ctx_id, snt, score in sentence_tuples:
            if ctx_id not in id_included:
                id_included.append(ctx_id)
                latest_ctxs.append(get_ctx_by_id[ctx_id])
        ex['snt_hybrid_rank'] = latest_ctxs

        if debug:
            for l,ctx in enumerate(latest_ctxs):
                if debug in ctx['text']:
                    print(f'/////  snt_hybrid_rank - {l}  /////')
                    print(ctx['text'])
                    break
            for l, tp in enumerate(sentence_tuples):
                if l<5 or debug in tp[1]:
                    print(f'#{l} {tp}')
            
        #####################################################################################################################
    end_time = time.time()
    duration = end_time - start_time
    duration /=len(examples)
    print(f"Hybrid Execution Time: {duration:.6f} seconds")

    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

    print(f'\nMetriever: {args.metriever_model} + {args.llm_name}')
    print('\n**** Answers ****\n')
    print('~~~~~~~ctx_keyword_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='ctx_keyword_rank', ans_key='answers')
    print('~~~~~~~ctx_semantic_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='ctx_semantic_rank', ans_key='answers')
    print('~~~~~~~snt_keyword_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='snt_keyword_rank', ans_key='answers')
    print('~~~~~~~snt_hybrid_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='snt_hybrid_rank', ans_key='answers') 

    print('\n**** Gold Evidences ****\n')
    print('~~~~~~~ctx_keyword_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='ctx_keyword_rank', ans_key='gold_evidences')
    print('~~~~~~~ctx_semantic_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='ctx_semantic_rank', ans_key='gold_evidences')
    print('~~~~~~~snt_keyword_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='snt_keyword_rank', ans_key='gold_evidences')
    print('~~~~~~~snt_hybrid_rank~~~~~~~~')
    eval_recall(examples_not_exact, ctxs_key='snt_hybrid_rank', ans_key='gold_evidences') 

    # save metriever results    
    save_name = f'./retrieved/{args.subset}_{args.stage1_model}_{args.stage2_model}_{args.metriever_model}_{args.llm_name}_outputs.json'
    if args.QFS_topk and args.QFS_topk>0:
        save_name = save_name.replace('_outputs', f'_qfs{args.QFS_topk}_outputs')
    if args.save_note:            
        save_name = save_name.replace('_outputs', f'_{args.save_note}_outputs')
    if debug==None and args.save==True:
        save_json_file(save_name, examples)
        print('Retrieved result saved. ', save_name)
    return


def separate_samples(examples):
    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = [], [], []
    for example in examples:
        if example['time_relation'] == '':
            examples_notime.append(example)
        elif int(example['exact']) == 1:
            examples_exact.append(example)
        else:
            examples_not_exact.append(example)
    return examples_notime, examples_exact, examples_not_exact


def get_spline_function(time_relation_type, implicit_condition, question_years):
    low = 0.6
    span = 50
    if len(question_years)==2:
        start = min(question_years)
        end = max(question_years)
    elif time_relation_type=='before':
        end = question_years[0]
        start = end-span    
    else:
        start = question_years[0]
        end = start+span

    x_points = np.array([start, end])
    if implicit_condition=='first':
        y_points = np.array([1, low])
    else:
        y_points = np.array([low, 1])
    linear_interp_function = interp1d(x_points, y_points, kind='linear')

    x_fine = np.linspace(start, end, 10)
    y_fine = linear_interp_function(x_fine)
    x_fine = [start-30, start-0.1] + list(x_fine) + [end+0.1, end+30]
    y_fine = [0.5, 0.5] + list(y_fine) + [0.5, 0.5]
    
    # # Plot the original points and the interpolated straight lines
    # # plt.plot(x_points, y_points, 'o', label='Original points')
    # plt.figure(figsize=(4, 3))
    # plt.plot(x_fine, y_fine)
    # plt.ylim(0.2, 1.1)
    # plt.yticks(np.arange(0.2, 1.1, 0.2))
    # # plt.xlabel('x')
    # # plt.ylabel('y')
    # title = ', '.join([str(s) for s in question_years])
    # plt.title(f'{implicit_condition} - {time_relation_type} - {title}')
    # # plt.grid(True, linestyle='-', color='lightgrey', axis='y')
    # plt.tight_layout()
    # plt.savefig('spline_plot.png')
    # import ipdb; ipdb.set_trace()

    return linear_interp_function


def get_temporal_coeffs(years, sentence_tuples, time_relation_type, implicit_condition, spline):
    temporal_coeffs = []
    for _, snt, _ in sentence_tuples:
        snt_years = year_identifier(snt)
        closest_year = None
        if time_relation_type == 'between':
            # between A and B
            start = min(years)
            end = max(years)
            if snt_years:
                relevant_years = [y for y in snt_years if y>=start and y<=end]
                if len(relevant_years)>0:
                    relevant_years = sorted(relevant_years)
                    if implicit_condition=='first':
                        closest_year = relevant_years[0]
                    else:
                        closest_year = relevant_years[-1]
        else:
            # before / after
            question_year = years[0]
            if snt_years:
                if time_relation_type == 'before':
                    relevant_years = [y for y in snt_years if y<=question_year]
                else:
                    relevant_years = [y for y in snt_years if y>=question_year]
                relevant_years = sorted(relevant_years)
                if len(relevant_years)>0:
                    if implicit_condition=='first':
                        closest_year = min(relevant_years)
                    else:
                        closest_year = max(relevant_years)
        # temporal score calculation
        try:
            coeff = spline(closest_year)
        except Exception:
            coeff = 0.5
        temporal_coeffs.append(coeff)
    return temporal_coeffs

# for SFR
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


if __name__ == "__main__":
    main()