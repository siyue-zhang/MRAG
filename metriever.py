from utils import *
from prompts import *
# import ipdb; ipdb.set_trace()
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams

from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker, FlagLLMReranker

from nltk.tokenize import sent_tokenize

debug = None
debug_question = None

# debug = 'He is best known as the drummer for American hard rock band Guns'
# debug_question = 'Who is the drummer for Guns and Roses after 2006?'

# debug = 'Krista White, Sophie Sumner, Jourdan Miller and India Gants from Cycles 14, 18 the "British Invasion", 20 "Guys & Girls" and 23 respectively.'
# debug_question = "Who wins America's Next Top Model Cycle 20 as of 2021?"

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
        choices=['metriever','minilm6','minilm12','bge','tinybert','bigegemma','monot5', None], 
        default='metriever', #
        # default='bge', #
        help='Choose a model for stage 2 re-ranking'
    )
    parser.add_argument(
        '--metriever-model', 
        choices=['minilm6','minilm12','bge','tinybert','bgegemma'], 
        default='minilm12',
        help='Choose a model for metriever stage2 re-ranking'
    )
    parser.add_argument('--contriever-output', type=str, default="./TempRAGEval/contriever_output/TempRAGEval.json")
    parser.add_argument('--bm25-output', type=str, default="./TempRAGEval/BM25_output/TempRAGEval.json")
    parser.add_argument('--ctx-topk', type=int, default=100)
    parser.add_argument('--QFS-topk', type=int, default=5)
    parser.add_argument('--snt-topk', type=int, default=200)
    parser.add_argument('--complete-ctx-text', type=bool, default=True)
    parser.add_argument('--hybrid-score', type=bool, default=True)
    parser.add_argument('--hybrid-base', type=float, default=0)
    parser.add_argument('--snt-with-title', type=bool, default=True)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--save-note', type=str, default=None)
    parser.add_argument('--subset', type=str, default='situatedqa')
    parser.add_argument('--not-save', type=bool, default=False)
    parser.add_argument('--load-keywords', type=bool, default=False)

    args = parser.parse_args()
    args.m1 = retrival_model_names(args.stage1_model)
    args.m2 = retrival_model_names(args.stage2_model) if args.stage2_model is not None else None
    args.m3 = retrival_model_names(args.metriever_model)
    args.l = llm_names(args.llm, instruct=True)
    args.llm_name = deepcopy(args.llm)

    # load llm
    if args.m2=='metriever':
        flg = '70b' in args.llm_name
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ", max_model_len=4096)
        else:
            args.llm = LLM(args.l, tensor_parallel_size=1, dtype='half', max_model_len=4096)
        
    # load semantic ranker for stage 2 / metriever
    if args.m2:
        name = args.m3 if args.m2 == 'metriever' else args.m2
        if 'gemma' in name:
            args.model = FlagLLMReranker(name, use_fp16=True,)
        elif 'bge' in name:
            args.model = FlagReranker(name, use_fp16=True)
        elif 'monot5' in name:
            pass
            # from pygaggle.rerank.base import Query, Text
            # from pygaggle.rerank.transformer import MonoT5
            # import os
            # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            # return MonoT5()
        else:
            args.model = CrossEncoder(name)

    # load examples
    if args.stage1_model == 'contriever':
        ctx_key = 'ctxs'
        path = args.contriever_output
        examples = load_contriever_output(path)
    elif args.stage1_model == 'bm25':
        ctx_key = 'bm25_ctxs'
        path = args.bm25_output
        with open(path, 'r') as file:
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

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]

    if debug_question:
        examples = [ex for ex in examples if ex['question']==debug_question]

    # temporary
    for ex in examples:
        if ' \'s ' in ex['question']:
            ex['question'] = ex['question'].replace(' \'s ', '\'s ')
    # examples = examples[300:365]

    # only keep situatedqa and timeqa samples for this code
    if args.subset == 'timeqa':
        examples = [ex for ex in examples if ex["source"] == 'timeqa']
        print('\nkeep only TimeQA subset.')
    elif args.subset == 'situatedqa':
        examples = [ex for ex in examples if ex["source"] == 'situatedqa']
        print('\nkeep only SituatedQA subset.')

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
                if ctx_id not in complete_ctx_map:
                    page = wiki[ctx['title']]
                    flgs = [p['id'] == ctx_id for p in page]
                    if any(flgs)==True and flgs.index(True)>0:
                        prev = page[flgs.index(True)-1]
                        ctx_sentences = sent_tokenize(prev['text'])
                        ctx_sentences_clean = [s.strip() for s in ctx_sentences]
                        text = ctx_sentences_clean[-1] + ' ' + ctx['text'].strip()
                    else:
                        text = ctx['text']
                    complete_ctx_map[ctx_id] = text
                ctx['text'] = complete_ctx_map[ctx_id]
                complete_ctxs.append(ctx)
            ex[ctx_key] = complete_ctxs


    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

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
        if args.stage2_model=='monot5':
            pass
            # for example in examples:
            #     query = Query(example["question"])
            #     passages = example[ctx_key][:100]
            #     texts = [Text(p["text"], {"id": p["id"], "title": p["title"], "hasanswer": p["hasanswer"]}, p["score"]) for p in passages]
            #     reranked = model.rerank(query, texts)
            #     latest_ctxs = [{"id": ex.metadata["id"], "title": ex.metadata["title"], "text": ex.text, "score": ex.score, "hasanswer": ex.metadata["hasanswer"]} for ex in reranked]
            #     latest_ctxs = sorted(latest_ctxs, key=lambda x: x["score"], reverse=True)
            #     example['reranker_ctxs'] = latest_ctxs
        else:
            flg = 'bge' in args.stage2_model
            for ex in tqdm(examples, desc="Reranking contexts"):
                question = ex['question']
                latest_ctxs = deepcopy(ex[ctx_key])
                latest_ctxs = latest_ctxs[:args.ctx_topk]
                model_inputs = [[question, ctx["title"]+" "+ctx["text"]] for ctx in latest_ctxs]
                scores = args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)   
                for i, ctx in enumerate(latest_ctxs):
                    ctx["score"] = float(scores[i])
                latest_ctxs = sorted(latest_ctxs, key=lambda x: x['score'], reverse=True)
                ex['reranker_ctxs'] = latest_ctxs
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
    if args.load_keywords:
        def load_example_keywords(path='./outputs/tmp_get_keywords.json'):
            with open(path, 'r') as file:
                return json.load(file)
        examples, question_keyword_map = load_example_keywords()
        print('loaded keywords.')
    else:
        # prepare keywords
        question_keyword_map={}
        for k, ex in enumerate(tqdm(examples, desc="Preprocessing questions", total=len(examples))):
            question = ex['question']
            time_relation = ex['time_relation']
            assert time_relation in question, question
            if time_relation != '':
                parts = question.split(time_relation)
                no_time_question = time_relation.join(parts[:-1])
                date = parts[-1]
                years = year_identifier(date)
                ex['years'] = years # int
            else:
                no_time_question = question
            normalized_question, implicit_condition = remove_implicit_condition(no_time_question)
            ex['implicit_condition'] = implicit_condition
            normalized_question = normalized_question[:-1] if normalized_question[-1] in '.?!' else normalized_question
            ex['normalized_question'] = normalized_question
            if normalized_question not in question_keyword_map:
                question_keyword_map[normalized_question]=[]
            print(f'==={k}===')
            print('Question : ', question)
            print('Normalized Question : ', normalized_question)


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

    # main reranking loop
    print('\nfinished preparation, start modular reranking.')
    for k, ex in enumerate(examples):
        question = ex['question']
        time_relation = ex['time_relation']
        normalized_question = ex['normalized_question']
        expanded_keyword_list, keyword_type_list = question_keyword_map[normalized_question]
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
        sentence_tuples = []
        get_ctx_by_id = {}
        # generate summaries
        QFS_prompts = []
        for ctx in latest_ctxs[:args.QFS_topk]:
            # ctx_id = ctx['id']
            # # before/after index
            # page = wiki[ctx['title']]
            # pid = [p['id'] for p in page].index(ctx_id)
            # texts = [p['text'] for p in page]
            # ctx_bf = pid-1 if pid>0 else None
            # ctx_af = pid+1 if pid<len(page)-1 else None
            # ctext = ''
            # if ctx_bf:
            #     ctext += texts[ctx_bf] + ' '
            # ctext += ctx['text'] + ' '
            # if ctx_af:
            #     ctext += texts[ctx_af]
            # ctext = ctext.strip()
            ctext = ctx['text']
            QFS_prompts.append(get_QFS_prompt(normalized_question, ctx['title'], ctext))

        if args.llm_name != 'gpt':
            summary_responses = call_pipeline(args, QFS_prompts)
        else:
            raise NotImplemented

        

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
        # classify time relation
        # there are 4 types of time relation and 2 types of implicit condition (first and last)
        # 1)	before (before, as of, by until)
        # 2)	after (after, from, since)
        # 3)	between (between, from to)
        # 4)	other (in, on, around, during)
        if 'years' in ex:
            years = ex['years'] # question dates
            time_relation = ex['time_relation'].lower()
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

        # static = question_static_map[normalized_question]
        # compute semantic scores using reranker
        if 'years' in ex and time_relation_type != 'other' and args.hybrid_score:
            # for hybrid ranking using question without time
            model_inputs = [[normalized_question, tp[1]] for tp in sentence_tuples]
        else:
            # rank by date matching
            model_inputs = [[question, tp[1]] for tp in sentence_tuples]
        semantic_scores =  args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)
        semantic_scores = [float(s) for s in semantic_scores]

        # print(f'\nQuestion is static: {static}')
        if 'years' in ex and time_relation_type != 'other' and args.hybrid_score:
            # use temporal-semantic hybrid ranker
            years = ex['years']
            if time_relation=='from' and len(years)<2:
                time_relation='after'
            if len(years)>1:
                assert time_relation in ['between','from']
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
    if debug==None and args.not_save==False:
        save_json_file(save_name, examples)
        print('Retrieved result saved.')
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

    # x_fine = np.linspace(start, end, 10)
    # y_fine = linear_interp_function(x_fine)

    # # Plot the original points and the interpolated straight lines
    # plt.plot(x_points, y_points, 'o', label='Original points')
    # plt.plot(x_fine, y_fine, label='Linear interpolation')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Linear Interpolation')
    # plt.legend()
    # plt.savefig('spline_plot.png')

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


def call_pipeline(args, prompts):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
    outputs = args.llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    responses = [res.split('Question:')[0] if 'Question:' in res else res for res in responses]
    responses = [res.split('<doc>')[0] if '<doc>' in res else res for res in responses]
    responses = [res.split('</doc>')[0] if '</doc>' in res else res for res in responses]
    responses = [res.split('Note:')[0] if 'Note:' in res else res for res in responses]
    responses = [res.replace('\n','').strip() for res in responses]
    return responses 


if __name__ == "__main__":
    main()