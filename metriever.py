from utils import *
from prompts import *
# import ipdb; ipdb.set_trace()

import argparse
from copy import deepcopy
import torch
device1 = 'cuda:0'
device2 = 'cuda:1' if torch.cuda.device_count()>1 else device1

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker, FlagLLMReranker

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
    parser.add_argument('--QFS-topk', type=int, default=20)
    parser.add_argument('--snt-topk', type=int, default=200)
    parser.add_argument('--hybrid-score', type=bool, default=True)
    parser.add_argument('--hybrid-base', type=float, default=0.5)
    parser.add_argument('--snt-with-title', type=bool, default=True)
    parser.add_argument('--llm', type=str, default="llama_8b")
    parser.add_argument('--step-by-step', type=bool, default=False)

    args = parser.parse_args()
    args.m1 = retrival_model_names(args.stage1_model)
    args.m2 = retrival_model_names(args.stage2_model) if args.stage2_model is not None else None
    args.m3 = retrival_model_names(args.metriever_model)
    args.l = llm_names(args.llm)

    if args.m2:
        if args.m2 == 'metriever':
            # load llm for metriever
            args.llm.device = device2
            if args.llm == 'llama_70b':
                from awq import AutoAWQForCausalLM
                args.llm = AutoAWQForCausalLM.from_pretrained(
                    args.l,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map=device2,
                    )
                args.llm.tokenizer = AutoTokenizer.from_pretrained(args.l)
            else:
                args.llm = AutoModelForCausalLM.from_pretrained(
                    args.l,
                    torch_dtype="auto",  
                    device_map=device2,
                    trust_remote_code=True,  
                ) 
                args.llm.tokenizer = AutoTokenizer.from_pretrained(args.l)

    # load semantic ranker for stage 2 / metriever
    if args.metriever_model=='bgegemma':
        args.model = FlagLLMReranker(args.m3, use_fp16=True, device=device1)
    elif args.metriever_model=='bge':
        args.model = FlagReranker(args.m3, use_fp16=True, device=device1)
    else:
        args.model = CrossEncoder(args.m3, device=device1)

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

    # separate samples into different types for comparison
    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

    ctx_key = 'ctxs' if args.stage1_model=='contriever' else 'bm25_ctxs'
    if args.m2 == None or args.m2 != 'metriever':
        print('\n**** Answers ****')
        print(f'--- Stage 1: {args.stage1_model} ---')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='answers')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='answers')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='answers')

        print('\n**** Gold Evidences ****')
        print(f'--- Stage 1: {args.stage1_model} ---')
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
        for ex in examples:
            question = ex['question']
            latest_ctxs = deepcopy(ex[ctx_key])
            model_inputs = [[question, ctx["title"]+" "+ctx["text"]] for ctx in latest_ctxs]
            scores = args.model.compute_score(model_inputs) if 'bge' in args.metriever_model else args.model.predict(model_inputs)   
            for i, ctx in enumerate(latest_ctxs):
                ctx["score"] = float(scores[i])
            ex['reranker_ctxs'] = latest_ctxs
        # evaluate reranking results    
        ctx_key = 'reranker_ctxs'
        examples_notime, examples_exact, examples_not_exact = separate_samples(examples)

        print('\n\n')
        print('\n**** Answers ****')
        print(f'--- Stage 2: {args.stage2_model} ---')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='answers')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='answers')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='answers')

        print('\n**** Gold Evidences ****')
        print(f'--- Stage 2: {args.stage2_model} ---')
        print('w/o date')
        eval_recall(examples_notime, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ key date')
        eval_recall(examples_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        print('w/ perturb date')
        eval_recall(examples_not_exact, ctxs_key=ctx_key, ans_key='gold_evidences')
        # save baseline results    
        save_json_file(f'./retrieved/{args.stage1_model}_{args.stage2_model}_outputs.json', examples)
        return

    ##### Metriever #####

    # prepare keywords
    question_keyword_map={}
    for k, ex in enumerate(examples):
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

    pipeline = transformers.pipeline(
        "text-generation", model=args.llm, tokenizer=args.tokenizer, device_map=args.llm.device
    )

    prompts = [get_keyword_prompt(q) for q in question_keyword_map]
    map_q = [q for q in question_keyword_map]
    keywords = pipeline(prompts, max_new_tokens=100, num_return_sequences=1, temperature=0.1)
    keywords = [k[0]['generated_text'] for k in keywords]
    for i in range(len(prompts)):
        tmp = keywords[i][len(prompts[i]):].strip()
        tmp = tmp.split('Question:')[0].replace('\n','').strip()
        tmp = eval(tmp)
        tmp = expand_keywords(tmp, map_q[i])
        question_keyword_map[map_q[i]] = tmp
    


    for k, ex in enumerate(examples):
        question = ex['question']
        time_relation = ex['time_specifier']
        normalized_question = ex['normalized_question']
        expanded_keyword_list, keyword_type_list = question_keyword_map[normalized_question]
        latest_ctxs = deepcopy(ex['ctxs']) # start from contriever top 1000

        # top 1000 ctx_keyword_rank_module
        ctx_kw_scores=[]
        for ctx in latest_ctxs:
            text = ctx['title'] + ' ' + ctx['text']
            ctx_score = count_keyword_scores(text, expanded_keyword_list, keyword_type_list)
            ctx_kw_scores.append((ctx, ctx_score))
        ctx_kw_scores = sorted(ctx_kw_scores, key=lambda x: x[1], reverse=True)
        latest_ctxs = [tp[0] for tp in ctx_kw_scores[:args.ctx_topk]] # only keep top 100
        ex['ctx_keyword_rank'] = latest_ctxs

        # top 100 ctx_semantic_rank_module
        model_inputs = [[normalized_question, ctx["title"]+ ' ' + ctx["text"]] for ctx in latest_ctxs]
        scores = args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)  
        for i, ctx in enumerate(latest_ctxs):
            ctx["score"] = float(scores[i]) # update contriever score to reranker score
        latest_ctxs = sorted(latest_ctxs, key=lambda x: x["score"], reverse=True)
        ex['ctx_semantic_rank'] = latest_ctxs

        # top 200 snt_keyword_rank_module
        # add QFS summary for top semantic context
        sentence_tuples = []
        get_ctx_by_id = {}
        for idx, ctx in enumerate(latest_ctxs):
            get_ctx_by_id[ctx['id']] = ctx
            snts = sent_tokenize(ctx['text'])
            if args.snt_with_title:
                snts = [ctx['title']+' '+snt for snt in snts]
            if idx < args.QFS_topk:
                QFS_prompt = get_QFS_prompt(normalized_question, ctx['title'], ctx['text'])
                summary = pipeline(QFS_prompt, max_new_tokens=200, num_return_sequences=1, temperature=0.1)
                summary = summary[0]['generated_text']
                summary = summary[len(QFS_prompt):]
                summary = summary.split('Question:')[0].replace('\n','').strip()
                print(QFS_prompt)
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
        print(f'total {len(sentence_tuples)} sentences.')
        # get new ctx rank based on sentence rank 
        latest_ctxs = []
        id_included = []
        for ctx_id, snt, score in sentence_tuples:
            if ctx_id not in id_included:
                id_included.append(ctx_id)
                latest_ctxs.append(get_ctx_by_id[ctx_id])
        ex['snt_keyword_rank'] = latest_ctxs
        
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

        # compute semantic scores using reranker
        if 'years' in ex and time_relation_type != 'other' and args.hybrid:
            # for hybrid ranking using question without time
            model_inputs = [[normalized_question, tp[1]] for tp in sentence_tuples]
        else:
            # rank by date matching
            model_inputs = [[question, tp[1]] for tp in sentence_tuples]
        semantic_scores =  args.model.compute_score(model_inputs) if flg else args.model.predict(model_inputs)
        semantic_scores = [float(s) for s in semantic_scores]

        if 'years' in ex and time_relation_type != 'other' and args.hybrid:
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

        latest_ctxs = []
        id_included = []
        for ctx_id, snt, score in sentence_tuples:
            if ctx_id not in id_included:
                id_included.append(ctx_id)
                latest_ctxs.append(get_ctx_by_id[ctx_id])
        ex['snt_hybrid_rank'] = latest_ctxs

    examples_notime, examples_exact, examples_not_exact = separate_samples(examples)
    if args.step_by_step:
        print('\n**** Answers ****\n')
        print('~~~~~~~ctx_keyword_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='ctx_keyword_rank', ans_key='answers')
        print('~~~~~~~ctx_semantic_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='ctx_semantic_rank', ans_key='answers')
        print('~~~~~~~snt_keyword_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='snt_keyword_rank', ans_key='answers')
        print('~~~~~~~snt_hybrid_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='snt_hybrid_rank', ans_key='answers') 

        print('\n**** Gold Evidences ****\n')
        print('~~~~~~~ctx_keyword_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='ctx_keyword_rank', ans_key='gold_evidences')
        print('~~~~~~~ctx_semantic_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='ctx_semantic_rank', ans_key='gold_evidences')
        print('~~~~~~~snt_keyword_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='snt_keyword_rank', ans_key='gold_evidences')
        print('~~~~~~~snt_hybrid_rank~~~~~~~~')
        eval_recall(examples_notime, ctxs_key='snt_hybrid_rank', ans_key='gold_evidences') 

    # save baseline results    
    save_json_file(f'./retrieved/{args.stage1_model}_{args.stage2_model}_{args.metriever_model}_outputs.json', examples)
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



if __name__ == "__main__":
    main()