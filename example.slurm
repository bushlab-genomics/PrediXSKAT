#!/bin/bash

#SBATCH -J skatmods
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH -o /slurm/logs/out_%A_%a_%j
#SBATCH --mail-user=email
#SBATCH --mail-type=ALL
#SBATCH --array=1-25

module purge
module load Miniconda3
eval "$(~/mambaforge/bin/conda shell.bash hook)"
conda activate polars

echo "skat activated"

config=config.txt
causalp=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
positivep=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
output=${SLURM_ARRAY_JOB_ID}_01_nulleffect.csv

echo $output
python -u p01_nulleffect.py --causal $causalp --positive $positivep --output $output

echo "script complete"