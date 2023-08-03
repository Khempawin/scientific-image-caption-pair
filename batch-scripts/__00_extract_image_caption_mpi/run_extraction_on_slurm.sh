#!/bin/bash

#SBATCH --nodes=<number of nodes to execute task>
#SBATCH --ntasks=<number of tasks>
#SBATCH --partition=<name of partition type>
#SBATCH --cpus-per-task=<number of cpus allocated per task>
#SBATCH --time=<time limit on task : e.g. 20:00:00 -> 20 hours >
#SBATCH --output=<stdout filename : relative to directory of scheduling : e.g. stdout-%f.out>

# load necessary modules
module purge
module load python/3.10.2
module load gcc/11.2.0 openmpi

export SLURM_EXPORT_ENV=ALL

# activate virtual environment if necessary
# . /path/to/venv/bin/activate
mpirun -n 16 python3 extract_image_caption.py -i <input directory> -o <output directory> -n <number of workers> -l <log level: info | debug> -flatten <flatten output directory or not, should use False> --omit_image_file <Use to set to output image files or not [True | False]> --caption_output_type <Type of output file [csv | parquet]>
# deactivate virtual environment if necessary
# deactivate