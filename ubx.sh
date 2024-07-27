#!/bin/bash

#SBATCH --job-name="ps"
#SBATCH --time=23:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1


module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

wandb login --relogin

python -m pip install pip --upgrade
rm -rf venv

sh ./install.sh

source ~/33_polar_segment/venv/bin/activate

python -m pip install torch torchvision torchaudio

python -c "import torch; print(torch.cuda.is_available())"

sh ./train_stack.sh
