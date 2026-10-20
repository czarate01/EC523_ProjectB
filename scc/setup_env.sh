#!/bin/bash -l
# One-time setup on an SCC login node. Run from the project root:
#     bash scc/setup_env.sh /projectnb/<project>/<team>/env
set -euo pipefail
ENV_PATH=${1:?usage: bash scc/setup_env.sh /projectnb/<project>/<team>/env}

module load miniconda
conda create -y -p "$ENV_PATH" python=3.11
conda activate "$ENV_PATH"
pip install -r cli/requirements.txt   # pinned: torch 2.6.0, pesq 0.0.4, pystoi 0.4.1, ...
python -c "import torch; print('torch', torch.__version__)"
