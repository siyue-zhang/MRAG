#!/bin/bash
#SBATCH --job-name=lmf20
#SBATCH --open-mode=append
#SBATCH --output=./monitor/lmf20.out
#SBATCH --error=./monitor/tlmf20.err
#SBATCH --export=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& export VLLM_WORKER_MULTIPROC_METHOD=spawn \
&& conda activate /scratch/sz4651/Projects/ext3/miniconda3/envs/metriever \
&& cd /scratch/sz4651/Projects/metriever_final \
&& python reader_final3.py --retriever-output situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json --paradigm fusion --reader llama --ctx-topk 20"



# && python reader_final2.py --retriever-output situatedqa_contriever_minilm12_outputs.json --paradigm concat --reader timo"
# && python reader_final2.py --retriever-output situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json --ctx-key-s2 snt_hybrid_rank"
