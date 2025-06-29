#!/bin/bash
#SBATCH --job-name=bm25
#SBATCH --open-mode=append
#SBATCH --output=./monitor/bm25.out
#SBATCH --error=./monitor/bm25.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=200G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/xxx/Projects/UnifiedSKG/overlay-skg2.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "
source ~/.bashrc \
&& conda activate /scratch/xxx/Projects/ext3/miniconda3/envs/metriever \
&& cd /scratch/xxx/Projects/metriever_final/bm25 \
&& python retrieve_bm25.py"

