#!/bin/bash
#SBATCH --job-name=reader
#SBATCH --open-mode=append
#SBATCH --output=./monitor/reader.out
#SBATCH --error=./monitor/reader.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=200G
#SBATCH -c 2

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& export VLLM_WORKER_MULTIPROC_METHOD=spawn \
&& source activate /scratch/sz4651/miniconda3/envs/l3 \
&& cd /scratch/sz4651/Projects/metriever_final \
&& python reader.py"

