import json
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

import sys
sys.path.append('../')
from contriever.src.evaluation import SimpleTokenizer, has_answer

lemmatizer = WordNetLemmatizer()
number_map = {
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen',
    '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}
number_map_b = {number_map[k]: k for k in number_map}
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
    return m

def llm_names(l):
    if l == "llama_8b":
        l = "meta-llama/Meta-Llama-3.1-8B"
    elif l== "llama_70b":
        l = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    elif l== "phi":
        l = "microsoft/Phi-3.5-mini-instruct"
    return l

def load_contriever_output(path):
    examples = []
    with open(path, 'r') as file:
        print(f'loaded: {path}')
        for line in file:
            examples.append(json.loads(line))
    return examples


def remove_implicit_condition(no_time_question):
    mapping={
        'last time': 'time',
        'the latest': 'the',
        'first time': 'time',
        'earliest time': 'time',
        'the first': '',
        'the last': '',
        'latest': '',
        'most recent': '',
        'recent':'',
    }
    mapping_type={
        'last time': 'last',
        'the latest': 'last',
        'first time': 'first',
        'earliest time': 'first',
        'the first': 'first',
        'the last': 'last',
        'latest': 'last',
        'most recent': 'last',
        'recent':'last',
    }
    implicit_condition = None
    for key in mapping:
        if key in no_time_question:
            no_time_question = no_time_question.replace(key, mapping[key])
            implicit_condition = mapping_type[key]
            break
    no_time_question = no_time_question.strip()
    if no_time_question.endswith(" right"):
        no_time_question = no_time_question[:-6]
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
        new_kw = None
        if kw[0].isupper():
            keyword_types[kw] = 'special'
        elif kw.isdigit():
            keyword_types[kw] = 'numeric'
            if kw in number_map:
                new_kw = number_map[kw]
        elif kw.lower() in number_map_b:
            keyword_types[kw] = 'numeric'
            new_kw = number_map_b[kw.lower()]
        else:
            n_words = len(kw_list)
            index = None
            for i in range(len(q_words)):
                flgs = []
                for j in range(len(kw_list)):
                    flgs.append(q_words[i+j].lower()==kw_list[j].lower())
                if all(flgs):
                    index=i
                    break
            assert index is not None
            last_word = kw_list[-1]
            last_index = index + n_words -1
            last_tag = q_tags[last_index]
            if last_tag.startswith('J'):
                keyword_types[kw] = 'superlative' if last_word[-3:]=='est' or last_word.lower()=='most' else 'adjective'
            else:
                keyword_types[kw] = 'general'
                last_lemma = q_lemmas[last_index]
                new_kw = kw.replace(last_word, last_lemma)

        tmp = [kw, new_kw] if new_kw else [kw]
        tmp = list(set(tmp))
        expanded_keyword_list.append(tmp)
        keyword_type_list.append(keyword_types[kw])

    if verbose:
        print('after: ')
        print(expanded_keyword_list)
        print(keyword_type_list)
        print('\n')

    return expanded_keyword_list, keyword_type_list


def year_identifier(timestamp):
    pattern = r'\b\d{4}\b'
    years = re.findall(pattern, timestamp)
    if len(years)>0:
        years = [int(y) for y in years]
        years = list(set(years))
    else:
        years = None     
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


def save_json_file(path, object):
   with open(path , "w") as json_file:
        json.dump(object, json_file)