#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=atesting
#SBATCH --cpus-per-task=1
#SBATCH --job-name=mpi-test
#SBATCH --time=00:05:00
#SBATCH --output=mpi-test-%j.out

# Load necessary modules
module purge
module load python/3.10.2
module load gcc/11.2.0 openmpi

export SLURM_EXPORT_ENV=ALL

# Activate virtual env if necessary
# . /path/to/venv/bin/activate

mpirun -n $SLURM_NTASKS python compute_context_length.py --input_dir <directory containing parquet files> --output_dir <output base directory> --image_out False

# deactivate virtual environment if necessary
# deactivate