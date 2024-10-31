#!/bin/bash
#SBATCH --job-name=seq10
#SBATCH --open-mode=append
#SBATCH --output=./monitor/seq10.out
#SBATCH --error=./monitor/seq10.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=200G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& export VLLM_WORKER_MULTIPROC_METHOD=spawn \
&& conda activate /scratch/sz4651/Projects/ext3/miniconda3/envs/metriever \
&& cd /scratch/sz4651/Projects/metriever_final \
&& python reader_seq2.py --ctx-topk 10"


# && python reader.py --retriever-output timeqa_contriever_minilm12_outputs.json \
# --ctx-key-s2 reranker_ctxs"

