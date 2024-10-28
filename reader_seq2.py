
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from utils import *
from prompts import *
from metriever import call_pipeline

import pandas as pd
# import ipdb; ipdb.set_trace()

import argparse
from copy import deepcopy
from vllm import LLM, SamplingParams
 
from temp_eval import normalize


def checker(question, context):

    prompt = f"""Can you answer the question based on the context paragraph? Response Yes or No.
<Context>
{context}
</Context>
<Question>
{question}
</Question>
<Response>
"""
    return prompt

def decontext(title, text):

    prompt = f"""Your task is to rewrite the context paragraph into independent sentences.
Requirements are follows:
- Write one sentence per line.
- Each sentence should standalone with complete information: the specific subject, object, relation, and action.
- Each sentence should talk about only one thing or event.
- Each sentence should include the year and the date if it is mentioned or can be inferred.

There are some examples for you to refer to:

<Title>
List of international organization leaders in 2007
<\Title>
<Context>
Prohibition of Chemical Weapons (OPCW) ; Director-General - Rogelio Pfirter, Argentina (2002–present) ; Organization of American States ; Secretary-General - José Miguel Insulza, Chile (2005–7) ; Organisation of the Islamic Conference ;
</Context>
<Sentences>
- Rogelio Pfirter, Argentina is the Director-General of the Organisation for the Prohibition of Chemical Weapons (OPCW) from 2002 to the present.
- José Miguel Insulza, Chile is the Secretary-General of the Organization of American States from 2005 to 2007.
- Organisation of the Islamic Conference.
</Sentences>

<Title>
1941 World Series
<\Title>
<Context>
This was the first Subway Series between the Brooklyn Dodgers and New York Yankees (though the Yankees had already faced the crosstown New York Giants five times). These two teams would meet a total of seven times from 1941 to 1956 — the Dodgers' only victory coming in 1955 — with an additional five matchups after the Dodgers left for Los Angeles, most recently in 2024.
</Context>
<Sentences>
- The 1941 World Series was the first Subway Series between the Brooklyn Dodgers and the New York Yankees.
- The New York Yankees had already faced the crosstown New York Giants five times prior to the 1941 World Series.
- The Dodgers and Yankees would meet a total of seven times from 1941 to 1956.
- The Dodgers' only victory in the matchups between the Dodgers and Yankees came in 1955.
- There were an additional five matchups between the Dodgers and Yankees after the Dodgers left for Los Angeles.
- The most recent matchup between the Dodgers and Yankees occurred in 2024.
</Sentences>

Now your context paragraph and question are as follows.
<Title>
{title}
<\Title>
<Context>
{text}
</Context>
<Sentences>
"""
    return prompt

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


# prompt = decontext('List of international organization leaders in 2015', 'Secretary-General - Ronald Noble, United States (2000–present) ; President - Mireille Ballestrazzi, France (2012–present) ; International Federation of Red Cross and Red Crescent Societies ; President - Tadateru Konoé, Japan (2009–present) ; International Maritime Organization ; Secretary-General - Koji Sekimizu, Japan (2012–present) ; International Organization for Migration (IOM) ; Director-general - William Lacy Swing, United States (2008–present) ; International Telecommunication Union ; Secretary-General - Hamadoun Touré, Mali (2007–present) ; Organisation for the Prohibition of Chemical Weapons (OPCW) ; Director-General - Ahmet Üzümcü, Turkey (2010–present) ; Organization of the Petroleum Exporting Countries (OPEC) ; Secretary-General – Abdallah Salem el-Badri, Libya (2007–2016) ; Universal Postal Union ; Director-General - Édouard Dayan, France (2005–present) ; World Intellectual Property Organization (WIPO) ; Director-General - Francis Gurry (2008–present)')
prompt = decontext("Old-Timers' Day","In 1965, Joe DiMaggio hit a grand slam into the left field stands. In 1975, the Yankees held Old Timers' Day at Shea Stadium and prior to the game it was announced that Billy Martin had been hired as Yankees' manager for the first time. In 1978 Martin was re-hired on Old Timers' Day. In 1998, the Yankees celebrated the 20th anniversary of the 1977, 1978 and 1981 World Series that they played against the Los Angeles Dodgers, and invited some members of those Dodger teams. The game was won on a home run by Willie Randolph against Tommy John, who played in all three of those World Series, for the Dodgers in 1977 and 1978 and for ")
print(call_pipeline(args, [prompt], 500)[0])

