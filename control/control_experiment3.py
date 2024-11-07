from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker, FlagLLMReranker

import sys
sys.path.append('./contriever/')
from src.contriever import Contriever

ranker_name = 'minilm'
g = 'gemma' in ranker_name
if g:
    ranker = FlagReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
else:
    ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


# sentences = ['Most home runs in a season after March 23, 1962?',
#              'M&M Boys broke the single-season record for most home runs in 1958.',
#              'M&M Boys broke the single-season record for most home runs in 1959.',
#              'M&M Boys broke the single-season record for most home runs in 1960.',
#              'M&M Boys broke the single-season record for most home runs in 1961.',
#              'M&M Boys broke the single-season record for most home runs in 1962.',
#              'M&M Boys broke the single-season record for most home runs in 1963.',
#              'M&M Boys broke the single-season record for most home runs in 1964.',
#              'M&M Boys broke the single-season record for most home runs in 1965.',
#              'M&M Boys broke the single-season record for most home runs on March 23, 1959.',
#              'M&M Boys broke the single-season record for most home runs.',
#              ]

sentences = ['Beginning in 1923, Johnson served in the Colorado House of Representatives for four terms. He was lieutenant governor from 1931 to 1933. He represented Colorado for three terms in the United States Senate from 1937 until 1955, during which time from 1937 to 1940 he was an intraparty critic of the New Deal policies of U.S. President Franklin D. Roosevelt. Johnson served as the 26th and 34th governor of Colorado from January 10, 1933 until January 1, 1937 and from January 12, 1955 until January 8, 1957. He opposed FDR‚Äôs New Deal policies.',
             'Edwin C. Johnson took which position in 1931?',
             'Edwin C. Johnson took which position in 1932?',
             'Edwin C. Johnson took which position in 1933?',
             'Edwin C. Johnson took which position in 1935?',
             'Edwin C. Johnson took which position as of March 14, 1936?',
             'Edwin C. Johnson took which position as of 1936?',
             'Edwin C. Johnson took which position in 1936?',
             'Edwin C. Johnson took which position in 1938?',
             'Edwin C. Johnson took which position in 1939?',
             'Edwin C. Johnson took which position in 1940?',
             'Edwin C. Johnson took which position in 1945?',
             'Edwin C. Johnson took which position from January 10, 1933 to January 1, 1937?',
             'Edwin C. Johnson took which position in 1956?',
             'Edwin C. Johnson took which position in 1957?',
             'Edwin C. Johnson took which position in 1958?',
             'Edwin C. Johnson took which position in 1959?',
             'Edwin C. Johnson took which position in 1960?',
             ]

model_inputs = [[s, sentences[0]] for i, s in enumerate(sentences) if i>0]
if g:
    scores = ranker.compute_score(model_inputs)
else:
    scores = ranker.predict(model_inputs)

print('\n\n')
for r in range(len(model_inputs)):
    print(model_inputs[r][0])
    print(model_inputs[r][1])
    print(scores[r],'\n--')

# #### Contriever #####
# contriever = Contriever.from_pretrained("facebook/contriever") 
# tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:
# inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# embeddings = contriever(**inputs)

# print('\n\n')

# for i in range(1,len(sentences)):
#     print(sentences[0])
#     print(sentences[i])
#     score = embeddings[0] @ embeddings[i]
#     print(round(score.item(),5))
#     print('\n--')

