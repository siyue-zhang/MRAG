import json
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections

from vllm import LLM, SamplingParams

# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from pattern.en import lexeme

import sys
sys.path.append('../')
from contriever.src.evaluation import SimpleTokenizer, has_answer

from openai import OpenAI
# client = OpenAI()

EXCL = ['time', 'years', 'for', 'new', 'recent', 'current', 'whom', 'who', 'out', 'place', 'not']

lemmatizer = WordNetLemmatizer()
number_map = {
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
    '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
    '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth', '9th': 'ninth',
}
number_map_b = {number_map[k]: k for k in number_map}

month_to_number = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12
}

short_month_to_number = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

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
            
tokenizer = SimpleTokenizer()

def retrival_model_names(m):
    if m == 'minilm6':
        m = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    elif m == 'minilm12':
        m = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    elif m == 'tinybert':
        m = "cross-encoder/ms-marco-TinyBERT-L-6"
    elif m == 'bge':
        m = 'BAAI/bge-reranker-large'
    elif m == 'bgegemma':
        m = 'BAAI/bge-reranker-v2-gemma'
    elif m=='nv':
        m = 'nvidia/NV-Embed-v1'
    elif m=='nv2':
        m = 'nvidia/NV-Embed-v2'
    elif m=='electra':
        m = 'cross-encoder/ms-marco-electra-base'
    elif m=='jina':
        m = 'jinaai/jina-reranker-v2-base-multilingual'
    elif m=='sfr':
        m = 'Salesforce/SFR-Embedding-Mistral'
    return m

def llm_names(l, instruct=False):
    if l == "llama_8b":
        if instruct:
            l = "meta-llama/Llama-3.1-8B-Instruct"
        else:
            l = "meta-llama/Meta-Llama-3.1-8B"
    elif l== "llama_70b":
        l = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif l== "phi":
        l = "microsoft/Phi-3.5-mini-instruct"
    elif l=="timo":
        l="Warrieryes/timo-13b-hf"
    elif l=="timellama":
        l="chrisyuan45/TimeLlama-7b-chat"
    return l

def load_contriever_output(path):
    examples = []
    with open(path, 'r', encoding="utf-8") as file:
        print(f'loaded: {path}')
        for line in file:
            examples.append(json.loads(line))
    return examples


def remove_implicit_condition(no_time_question):
    mapping_type={
        ' latest': 'last',
        ' last': 'last',
        ' first': 'first',
        ' earliest': 'first',
        ' first': 'first',
        ' most recent': 'last',
        ' recent':'last',
    }
    implicit_condition = None
    for key in mapping_type:
        if key in no_time_question:
            no_time_question = no_time_question.replace(key, '')
            implicit_condition = mapping_type[key]
            break
    no_time_question = no_time_question.strip()
    if no_time_question[-1] not in '.?':
        no_time_question+='?'
    return no_time_question, implicit_condition


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun

def pattern_stopiteration_workaround():
    try:
        print(lexeme('gave'))
    except:
        pass
# Basically, the pattern code will fail the first time you run it, so you first need to run it once and catch the Exception it throws.
pattern_stopiteration_workaround()

def expand_keywords(keyword_list, normalized_question, verbose=False):
    if verbose:
        print('before expand: \n', keyword_list)

    q_words = []
    q_tags = []
    q_lemmas = []
    expanded_keyword_list = []
    keyword_type_list = []

    tokens = word_tokenize(normalized_question)
    tagged_tokens = pos_tag(tokens)
    for word, tag in tagged_tokens:
        q_words.append(word)
        q_tags.append(tag)
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        q_lemmas.append(lemma)

    keyword_dict={kw:[] for kw in keyword_list}
    keyword_types={}
    for kw in keyword_dict:
        kw_list = kw.split()
        new_kw = []
        if kw[0].isupper():
            keyword_types[kw] = 'special'
        elif kw.lower() in number_map:
            keyword_types[kw] = 'numeric'
            new_kw.append(number_map[kw.lower()])
        elif kw.lower() in number_map_b:
            keyword_types[kw] = 'numeric'
            new_kw.append(number_map_b[kw.lower()])
        else:
            n_words = len(kw_list)
            index = None
            try:
                for i in range(len(q_words)):
                    flgs = []
                    for j in range(len(kw_list)):
                        flgs.append(q_words[i+j].lower()==kw_list[j].lower())
                    if all(flgs):
                        index=i
                        break
            except Exception:
                pass
            if index:
                last_word = kw_list[-1]
                last_index = index + n_words -1
                last_tag = q_tags[last_index]
                if last_tag.startswith('J'):
                    keyword_types[kw] = 'superlative' if last_word[-3:]=='est' or last_word.lower()=='most' else 'adjective'
                else:
                    keyword_types[kw] = 'general'
                    new_kw += [kw.replace(last_word, x) for x in lexeme(last_word)]
            else:
                keyword_types[kw] = 'general'

        tmp = list(set([kw] + new_kw))
        if keyword_types[kw] == 'special' and ' and ' in kw:
            tmp += [kw.replace(' and ','&'), kw.replace(' and ', ' & '), kw.replace(' and ', ' N\' ')]
        if '-' in kw:
            tmp.append(kw.replace('-',' '))
        expanded_keyword_list.append(tmp)
        keyword_type_list.append(keyword_types[kw])

    if verbose:
        print('after: ')
        print(expanded_keyword_list)
        print(keyword_type_list)
        print('\n')

    return expanded_keyword_list, keyword_type_list


def replace_dates(text):
    # This regex captures date ranges in the format "1990–93" or "1990-93"
    pattern = r'(\b\d{4})[–-](\d{2}\b)'
    
    # Replacement function that reconstructs the full year range
    def replace_func(match):
        start_year = match.group(1)
        end_year = start_year[:2] + match.group(2)  # Combine first two digits from start year with the last two from the end year
        return ' '.join([str(i) for i in range(int(start_year),int(end_year)+1)])
        # return f"{start_year} {end_year}"
    
    # Apply the replacement
    return re.sub(pattern, replace_func, text)


def expand_year_range(text):
    # Define a function to replace a range with full years
    def replace_range(match):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        # Generate a space-separated string of all years in the range
        return ' '.join(str(year) for year in range(start_year, end_year + 1))

    # Regular expression to match both en-dash and hyphen date ranges
    pattern = r'(\d{4})[–-](\d{4})'

    # Replace the ranges using the replace_range function
    return re.sub(pattern, replace_range, text)


def year_identifier(timestamp):
    # replace "1990–93" by "1990" and "1993"
    timestamp = replace_dates(timestamp)
    timestamp = expand_year_range(timestamp)

    pattern = r'\b(\d{4})(?:s)?\b'
    years = re.findall(pattern, timestamp)
    
    if years:
        # Convert to integers, remove duplicates, and sort the years
        years = sorted(set(map(int, years)))
    else:
        # If no years are found, return None
        return None
    
    return years


def count_keyword_scores(text, expanded_keyword_list, keyword_type_list):
    text=text.lower()
    score=0
    weights={'special':1,'superlative':0.7,'general':0.4,'numeric':0.5,'adjective':0.4}
    for i in range(len(expanded_keyword_list)):
        keywords = expanded_keyword_list[i]
        kw_type = keyword_type_list[i]
        flg = False
        if kw_type == 'general':
            for kw in keywords:
                if kw.lower() in text:
                    flg = True
        else:
            flg = has_answer(keywords, text, tokenizer)
        if flg:
            score += weights[kw_type]
    return score


def get_recall(first_hits, num_pages):
    flgs=[]
    for x in first_hits:
        if x>-1 and x<num_pages:
            flgs.append(1)
        else:
            flgs.append(0)
    return np.mean(flgs)


def eval_recall(examples, ctxs_key, ans_key='answers'):
    first_hits=[]
    for example in tqdm(examples, desc="Processing Examples"):
        answer = example[ans_key]          
        has_answer_list = []
        cap = min(1000,len(example[ctxs_key]))
        for i, ctx in enumerate(example[ctxs_key][:cap]):
            text = ctx['title'] + ' ' + ctx['text']
            hasanswer_ctx = has_answer(answer, text, tokenizer)
            has_answer_list.append(hasanswer_ctx)
        first_hits.append(has_answer_list.index(True) if any(has_answer_list) else -1)
    records = [(f'#{i}', ft) for i, ft in enumerate(first_hits)]
    ver = []
    for k in [1,5,10,20,50,100]:
        recall = get_recall(first_hits, k)
        ver.append(f'R@{k:3} = {np.round(recall,4):4}')
    print(' | '.join(ver))
    print('\n')
    return records

def _str_f1(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.lower().strip().split()
  target_tokens = target.lower().strip().split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def max_token_f1(answers, prediction):
    f1 = [_str_f1(ans, prediction) for ans in answers]
    f1 = max(f1)
    return f1

def save_json_file(path, object):
   with open(path , "w") as json_file:
        json.dump(object, json_file)


def load_json_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def eval_reader(to_save, param_pred, subset='situatedqa', metric='acc'):

    to_save_situatedqa = [ex for ex in to_save if ex['source']==subset]
    exact_param, exact_rag = [], []
    not_exact_param, not_exact_rag = [], []
    for example in to_save_situatedqa:
        if example['time_relation'] == '':
            pass
        elif int(example['exact']) == 1:
            exact_rag.append(example[f'rag_{metric}'])
            if f'param_{metric}' in example:
                exact_param.append(example[f'param_{metric}'])
        else:
            not_exact_rag.append(example[f'rag_{metric}'])
            if f'param_{metric}' in example:
                not_exact_param.append(example[f'param_{metric}'])

    if param_pred:
        print('Parametric')
        print(f'    w/ key date {metric} : {round(np.mean(exact_param),4) if len(exact_param)>0 else 0}')
        print(f'    w/ perturb date {metric} : {round(np.mean(not_exact_param),4) if len(not_exact_param)>0 else 0}')
        print(f'    all date {metric} : {round(np.mean(not_exact_param + exact_param),4) if len(not_exact_param + exact_param)>0 else 0}')


    print('RAG')
    print(f'    w/ key date {metric} : {round(np.mean(exact_rag),4) if len(exact_rag)>0 else 0}')
    print(f'    w/ perturb date {metric} : {round(np.mean(not_exact_rag),4) if len(not_exact_rag)>0 else 0}')
    print(f'    all date {metric} : {round(np.mean(not_exact_rag + exact_rag),4) if len(not_exact_rag + exact_rag)>0 else 0}')


def check_no_knowledge(string):
    assert isinstance(string, str)
    flg=False
    if len(string)==0:
        flg=True
    else:
        string=string.lower()
        if 'not' in string.split():
            flg=True
        else:
            phrases = ['unknown', 'none', "unable", "no information", "no answer", "don't have"]
            flg = any([p in string for p in phrases])
    return flg

def force_string(item):
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        item =item[0] if len(item)>0 else ''
    try:
        item = str(item)
    except Exception as e:
        item = ''
    return item

import threading


def fetch_completion(client, prompt, timeout=10):
    """
    Fetch a completion from the API with a timeout mechanism.
    
    Args:
        client: The API client.
        prompt: The prompt to send to the API.
        timeout: The time (in seconds) to wait before retrying.
    
    Returns:
        The API response content or an error message if retries fail.
    """
    result = {"response": None}
    event = threading.Event()

    def api_call():
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result["response"] = completion.choices[0].message.content
        except Exception as e:
            result["response"] = f"Error: {e}"
        finally:
            event.set()

    # Start the API call in a separate thread
    thread = threading.Thread(target=api_call)
    thread.start()

    # Wait for the API call to finish or timeout
    event.wait(timeout)

    if not event.is_set():
        # Timeout occurred
        thread.join(0)  # Attempt to clean up the thread
        return "Timeout: API request took too long, retrying..."
    
    return result["response"]



def call_pipeline(args, prompts, max_tokens=100, return_list=False, ver=False):

    if args.reader and 'gpt' in args.reader.lower():

        responses = []
        for prompt in tqdm(prompts, desc="GPT"):
            # completion = client.chat.completions.create(
            #     model="gpt-4o-mini",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant that answers questions concisely."},
            #         {
            #             "role": "user",
            #             "content": prompt
            #         }
            #     ]
            # )
            # response = completion.choices[0].message.content
            # print(response)
            # responses.append(response.strip())

            for attempt in range(5):  # Retry up to 3 times
                response = fetch_completion(client, prompt, timeout=10)
                if "Timeout" not in response:
                    break
            print(response)
            responses.append(response.strip())

        print('\n\n')
        return responses


    if args.reader == None:
        sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=max_tokens, seed=0)
        outputs = args.llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        for stopper in ['</Keywords>', '</Summarization>', '</Answer>', '</Info>', '</Sentences>', '</Sentence>', '</Response>']:
            responses = [res.split(stopper)[0] if stopper in res else res for res in responses]
        return responses
    else:
        # if args.reader in ['timellama']:
        #     outputs = args.llm(prompts, do_sample=True, max_new_tokens=100, num_return_sequences=1, temperature=0.2, top_p=0.95)
        #     outputs = [r[0]['generated_text'] for r in outputs]
        #     responses = [outputs[i].replace(prompts[i],'') for i in range(len(prompts))]
        #     print('//////')
        #     print(prompts[0])
        #     print('//////')
        #     print(responses[0])
        #     import ipdb; ipdb.set_trace()
        # else:
        sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=max_tokens, seed=0)
        outputs = args.llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        if ver:
            for x, y in zip(prompts, responses):
                # print(x.split('Now your context paragraph and question are')[-1])
                print(x.split('Now your document and question are')[-1])
                print(y.split('</Summarization>')[0])
                # print(y.split('</Response>')[0])
                # print(y.split('</Answer>')[0])
                print('//////\n')
        # print(prompts[0])
        # print('//////')
        # print(responses[0])
        # import ipdb; ipdb.set_trace()

        for stopper in ['</Keywords>', '</Summarization>', '</Answer>', '</Info>', '</Sentences>', '</Sentence>', '</Response>']:
            responses = [res.split(stopper)[0] if stopper in res else res for res in responses]

        if return_list and '<Thought>' in prompts[0]:
            output = []
            for res in responses:
                res = res.split('\n</Answer>')[0]
                res = res.split('<Answer>\n')[-1]
                res = res.split('- ')
                res = [r.replace('\n','').strip() for r in res]
                res = [r for r in res if len(r)>0 ] # and len(r.split())<30
                output.append(res)
            return output
        elif '<Thought>' in prompts[0]:
            for mid_stopper in ['</Thought>', '<Answer>', '<Response>']:
                responses = [res.split(mid_stopper)[-1].replace('\n','').strip() if mid_stopper in res else res for res in responses]
        else:
            if return_list:
                responses = [res.split('- ') for res in responses]
                tmp = []
                for res in responses:
                    res = [r.replace('\n','').strip() for r in res]
                    res = [r for r in res if 'none' not in r.lower()]
                    tmp.append([r for r in res if r !=''])
                responses = tmp
            else:
                tmp=[]
                for r in responses:
                    ans = r.split('\n')
                    ans = [rr for rr in ans if len(rr)>0]
                    if len(ans)>0:
                        tmp.append(ans[0].strip())
                    else:
                        tmp.append('')
                responses = tmp
        return responses