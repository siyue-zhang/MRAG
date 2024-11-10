#!/bin/bash
#SBATCH --job-name=s70bt10
#SBATCH --open-mode=append
#SBATCH --output=./monitor/s70bt10.out
#SBATCH --error=./monitor/s70bt10.err
#SBATCH --export=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -c 2

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& export VLLM_WORKER_MULTIPROC_METHOD=spawn \
&& conda activate /scratch/sz4651/Projects/ext3/miniconda3/envs/metriever \
&& cd /scratch/sz4651/Projects/metriever_final \
&& python reader.py --retriever-output situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json --paradigm concat --reader llama_70b --ctx-topk 10 --param-cot True --param-pred True"



# && python reader_final2.py --retriever-output situatedqa_contriever_minilm12_outputs.json --paradigm concat --reader timo"
# && python reader_final2.py --retriever-output situatedqa_contriever_metriever_minilm12_llama_8b_qfs5_outputs.json --ctx-key-s2 snt_hybrid_rank"
