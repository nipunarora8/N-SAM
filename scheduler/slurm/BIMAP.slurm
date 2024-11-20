#!/bin/bash -l
#SBATCH --job-name=BIMAP-PLACEHOLDER_JOB_NAME	# Job name
#SBATCH -o ../logs/job_%x_%j.log
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00

export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

cd /home/hpc/iwb3/iwb3102h/bimap-dendrite-segmentation

module load python/3.10-anaconda

conda activate bimap

# Run python script
python PLACEHOLDER_JOB_NAME.py

conda deactivate
echo "BIMAP-PLACEHOLDER_JOB_NAME Job finished!!"