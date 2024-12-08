
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from FlagEmbedding import FlagLLMReranker
reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=True) # You can also set use_bf16=True to speed up computation with a slight performance degradation


# scores = reranker.compute_score([
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1958.'], 
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1959.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1960.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1961.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1962.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1963.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1964.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1965.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs on March 23, 1959.'],
#     ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs'],
#     ])
# print(scores)





# # from transformers import AutoTokenizer
# # from sentence_transformers import CrossEncoder
# from FlagEmbedding import FlagReranker, FlagLLMReranker

# # import sys
# # sys.path.append('./contriever/')
# # from src.contriever import Contriever

# ranker_name = 'bgegemma'
# g = 'gemma' in ranker_name
# if g:
#     ranker = FlagxxxReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
# # else:
# #     ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

query = 'Most home runs in a season before March 23, 1962?'
sentences = ['M&M Boys broke the single-season record for most home runs in 1958.',
             'M&M Boys broke the single-season record for most home runs in 1959.',
             'M&M Boys broke the single-season record for most home runs in 1960.',
             'M&M Boys broke the single-season record for most home runs in 1961.',
             'M&M Boys broke the single-season record for most home runs in 1962.',
             'M&M Boys broke the single-season record for most home runs in 1963.',
             'M&M Boys broke the single-season record for most home runs in 1964.',
             'M&M Boys broke the single-season record for most home runs in 1965.',
             'M&M Boys broke the single-season record for most home runs on March 23, 1959.',
             'M&M Boys broke the single-season record for most home runs.',
             ]

# # sentences = ['M&M Boys broke the single-season record for most home runs in 1961.',
# #              'Most home runs in a season before 2000?',
# #              'Most home runs in a season before 1990?',
# #              'Most home runs in a season before 1980?',
# #              'Most home runs in a season before 1970?',
# #              'Most home runs in a season before 1963?',
# #              'Most home runs in a season before 1962?',
# #              'Most home runs in a season before 1961?',
# #              'Most home runs in a season before 1960?',
# #              'Most home runs in a season before 1959?',
# #              'Most home runs in a season before 1950?',
# #              'Most home runs in a season before 1940?',
# #              'Most home runs in a season?',
# #              'Most home runs in a season before March 23, 1963?']

model_inputs = [[query,s] for s in sentences]
# if g:
scores = reranker.compute_score(model_inputs)
# else:
# scores = reranker.predict(model_inputs)

# print('\n\n')

for r in range(len(model_inputs)):
    print(model_inputs[r][0])
    print(model_inputs[r][1])
    print(scores[r],'\n--')

# # #### Contriever #####
# # contriever = Contriever.from_pretrained("facebook/contriever") 
# # tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:
# # inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# # embeddings = contriever(**inputs)

# # print('\n\n')

# # for i in range(1,len(sentences)):
# #     print(sentences[0])
# #     print(sentences[i])
# #     score = embeddings[0] @ embeddings[i]
# #     print(round(score.item(),5))
# #     print('\n--')

