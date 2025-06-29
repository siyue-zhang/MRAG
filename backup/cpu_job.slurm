#!/bin/bash
#SBATCH --job-name=x
#SBATCH --open-mode=append
#SBATCH --output=./monitor/x.out
#SBATCH --error=./monitor/x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH -c 1

singularity exec --nv --overlay /scratch/sz4651/Projects/UnifiedSKG/overlay-skg.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source ~/.bashrc \
&& source activate /scratch/sz4651/miniconda3/envs/contriever \
&& cd /scratch/sz4651/Projects/porqa/bm25 \
&& python /scratch/sz4651/Projects/porqa/bm25/convert_doc.py \
&& python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /scratch/sz4651/Projects/modular_retriever/bm25/index/enwiki-dec2021/input \
  --index /scratch/sz4651/Projects/modular_retriever/bm25/index/enwiki-dec2021/indexed/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw"


