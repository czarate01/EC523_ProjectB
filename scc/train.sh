#!/bin/bash -l
# Batch training job on the SCC. Submit from the project root, passing
# train.py arguments after the script name, e.g.
#     qsub -P <project> -N dccrn_e scc/train.sh \
#         --train-manifest DATA/train.csv --dev-manifest DATA/dev.csv --out runs/dccrn_e
# Re-submitting with the same --out resumes from runs/.../last.pt.
#
# Use `qgpus` to see GPU types; to measure memory/RTF on an L40S, add
# `-l gpu_type=L40S` to the qsub command.
#$ -l gpus=1
#$ -l gpu_c=8.6
#$ -pe omp 8
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd

ENV_PATH=${ENV_PATH:-/projectnb/ec523kb/students/czarate/env}
module load miniconda
conda activate "$ENV_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --workers 6 "$@"
