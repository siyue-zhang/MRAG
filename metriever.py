from utils import *
from prompts import *
# import ipdb; ipdb.set_trace()
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams

from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker, FlagLLMReranker

debug = None
debug_question = None

# debug = 'hitting a combined 115 home runs in 1961 and breaking the single-season record'
# debug_question = 'Which league did Albuquerque Dukes play for since 1972?'

# debug = 'Krista White, Sophie Sumner, Jourdan Miller and India Gants from Cycles 14, 18 the "British Invasion", 20 "Guys & Girls" and 23 respectively.'
# debug_question = "Who wins America's Next Top Model Cycle 20 as of 2021?"

def main():
    parser = argparse.ArgumentParser(description="Metriever")
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument(
        '--stage1-model',
        choices=['bm25', 'contriever'], 
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
    parser.add_argument('--contriever-output', type=str, default="./TempRAGEval/contriever_output/data.json")
    parser.add_argument('--bm25-output', type=str, default="./TempRAGEval/BM25_output/data.json")
    parser.add_argument('--ctx-topk', type=int, default=100)
    parser.add_argument('--QFS-topk', type=int, default=5)
    parser.add_argument('--snt-topk', type=int, default=200)
    parser.add_argument('--hybrid-score', type=bool, default=True)
    parser.add_argument('--hybrid-base', type=float, default=0.5)
    parser.add_argument('--snt-with-title', type=bool, default=True)
    parser.add_argument('--llm', type=str, default="llama_70b")
    parser.add_argument('--save-note', type=str, default=None)

    args = parser.parse_args()
    args.m1 = retrival_model_names(args.stage1_model)
    args.m2 = retrival_model_names(args.stage2_model) if args.stage2_model is not None else None
    args.m3 = retrival_model_names(args.metriever_model)
    args.l = llm_names(args.llm)
    args.llm_name = deepcopy(args.llm)

    # load llm
    if args.m2:
        flg = '70b' in args.llm_name
        if flg:
            args.llm = LLM(args.l, tensor_parallel_size=2, quantization="AWQ")
        else:
            args.llm = LLM(args.l, tensor_parallel_size=1, dtype='half', max_model_len=4096)
        
    # load semantic ranker for stage 2 / metriever
    if args.m2:
        name = args.m3 if args.m2 == 'metriever' else args.m2
        if 'gemma' in name:
            args.model = FlagLLMReranker(name, use_fp16=True,)
        elif 'bge' in name:
            args.model = FlagReranker(name, use_fp16=True)
        else:
            args.model = CrossEncoder(name)

    # load examples
    if args.stage1_model == 'contriever':
        path = args.contriever_output
        examples = load_contriever_output(path)
    else:
        path = args.bm25_output
        raise NotImplemented

    if args.max_examples:
        examples = examples[:min(len(examples),args.max_examples)]
    
    # only keep situatedqa and timeqa samples for this code
    examples = [ex for ex in examples if ex["source"] != 'dbpedia']
    if debug_question:
        examples = [ex for ex in examples if ex['question']==debug_question]

    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

    #####################################################################################################################
    # Baselines 

    ctx_key = 'ctxs' if args.stage1_model=='contriever' else 'bm25_ctxs'
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
        save_json_file(f'./retrieved/{args.stage1_model}_{args.stage2_model}_outputs.json', examples)
        return

    
    #####################################################################################################################
    # Metriever 

    # prepare keywords
    question_keyword_map={}
    for k, ex in enumerate(tqdm(examples, desc="Preprocessing questions", total=len(examples))):
        question = ex['question']
        time_relation = ex['time_specifier']
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

    print('\nstart extracting keywords using llm.')
    prompts = [get_keyword_prompt(q) for q in question_keyword_map]
    questions = [q for q in question_keyword_map]
    if args.llm_name not in ['gpt']:
        keyword_responses = call_pipeline(args, prompts)
    else:
        raise NotImplemented

    for i, q in enumerate(tqdm(questions, desc="Postprocessing keywords", total=len(questions))):
        tmp = eval(keyword_responses[i])
        revised = []
        for kw in tmp:
            while kw.lower() not in q.lower():
                # revise the extrcated keyword if not match with question
                kw = ' '.join(kw.split()[:-1])
            if kw!='' and kw.lower() in q.lower():
                revised.append(kw)
        revised = list(set(revised))
        revised = expand_keywords(revised, q, verbose=True)
        question_keyword_map[q] = revised

    # # judge if the question has static answer
    # question_static_map={}
    # prompts = [judge_static_prompt(q) for q in questions]
    # print('aaaaaa')
    # print(prompts)
    # if args.llm_name not in ['gpt']:
    #     responses = call_pipeline(args, prompts)
    # else:
    #     raise NotImplemented
    # for i, q in enumerate(tqdm(questions, desc="Postprocessing keywords (2)", total=len(questions))):
    #     question_static_map[q] = 'no' in responses[i].lower()
    # print(responses)
    # import ipdb; ipdb.set_trace()

    # main reranking loop
    print('\nfinished preparation, start modular reranking.')
    for k, ex in enumerate(examples):
        question = ex['question']
        time_relation = ex['time_specifier']
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
            QFS_prompts.append(get_QFS_prompt(normalized_question, ctx['title'], ctx['text']))

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
        # 1)	before
        # 2)	after
        # 3)	between
        # 4)	other (in, on, around)
        if 'years' in ex:
            years = ex['years'] # question dates
            time_relation = ex['time_specifier'].lower()
            implicit_condition = ex['implicit_condition']
            if time_relation in ['before','as of','by','until']:
                time_relation_type = 'before'
            elif time_relation == 'from':
                if len(years)==1:
                    time_relation_type = 'after'
                else:
                    time_relation_type = 'between'
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

        ex['top_snts'] = '\n\n'.join([tp[0] for tp in sentence_tuples[:20]])
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

    # save baseline results    
    save_name = f'./retrieved/{args.stage1_model}_{args.stage2_model}_{args.metriever_model}_{args.llm_name}_outputs.json'
    if args.save_note:
        save_name = save_name.replace('_outputs', f'_{args.save_note}_outputs')
    if debug==None:
        save_json_file(save_name, examples)
    return


def separate_samples(examples):
    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = [], [], []
    for example in examples:
        if example['time_specifier'] == '':
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
    responses = [res.replace('\n','').strip() for res in responses]
    return responses 


if __name__ == "__main__":
    main()