#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=atesting
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mpi-test
#SBATCH --time=00:05:00
#SBATCH --output=mpi-test-%j.out

# Load necessary modules
module purge
module load python/3.10.2
# module load gcc/11.2.0 openmpi

# export SLURM_EXPORT_ENV=ALL

# Activate virtual env if necessary
# . /path/to/venv/bin/activate

python select_data_with_dask.py -s0 <output directory of stage00> -s1 <output directory of stage01> -o <directory to save output parquet files + image files> -img <copy images from stage00 or not>

# deactivate virtual environment if necessary
# deactivate