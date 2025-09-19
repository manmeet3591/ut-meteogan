#!/bin/bash
#SBATCH -J meteogan_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -A ATM23017
#SBATCH -t 02:00:00
#SBATCH -o meteogan_train.o%j
#SBATCH -e meteogan_train.e%j

# Load Apptainer
module load tacc-apptainer

echo "Running on host $(hostname)"
echo "Starting at $(date)"

# Execute training script inside Apptainer container
apptainer exec --nv \
  --bind /scratch/08105/ms86336/meteogan:/opt/notebooks \
  /scratch/08105/ms86336/meteogan/meteogan.sif \
  python /opt/notebooks/train.py

echo "Finished at $(date)"
