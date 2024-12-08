from FlagEmbedding import FlagLLMReranker
reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=True) # You can also set use_bf16=True to speed up computation with a slight performance degradation


scores = reranker.compute_score([
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1958.'], 
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1959.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1960.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1961.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1962.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1963.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1964.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs in 1965.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs on March 23, 1959.'],
    ['Most home runs in a season after March 23, 1962?', 'M&M Boys broke the single-season record for most home runs'],
    ])
print(scores)
