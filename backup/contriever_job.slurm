#!/bin/bash
#SBATCH --job-name=TempRAGEval_
#SBATCH --open-mode=append
#SBATCH --output=./monitor/TempRAGEval_.out
#SBATCH --error=./monitor/TempRAGEval_.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -c 1



singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source ~/.bashrc \
&& source activate /scratch/sz4651/Projects/ext3/miniconda3/envs/contriever \
&& cd /scratch/sz4651/Projects/porqa/contriever \
&& python passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages /scratch/sz4651/Projects/porqa/wikipedia/enwiki-dec2021/psgs_w100.tsv \
    --passages_embeddings ""/scratch/sz4651/Projects/modular_retriever/enwiki-dec2021/contriever_embeddings/*"" \
    --n_docs 1000 \
    --data /scratch/sz4651/Projects/metriever_final/TempRAGEval/TempRAGEval.json \
    --output_dir /scratch/sz4651/Projects/metriever_final/TempRAGEval/contriever_output"

# singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
# source ~/.bashrc \
# && source activate /scratch/sz4651/Projects/ext3/miniconda3/envs/contriever \
# && cd /scratch/sz4651/Projects/porqa/contriever \
# && python generate_passage_embeddings.py \
#     --model_name_or_path facebook/contriever \
#     --output_dir /scratch/sz4651/Projects/modular_retriever/wikipedia_chunks/contriever_embeddings  \
#     --passages /scratch/sz4651/Projects/modular_retriever/wikipedia_chunks/psgs_qampari.tsv \
#     --shard_id 0 --num_shards 1"