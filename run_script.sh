#!/bin/bash

#SBATCH --nodes=<number of nodes to execute task>
#SBATCH --ntasks=<number of tasks>
#SBATCH --cpus-per-task=<number of cpus allocated per task>
#SBATCH --time=<time limit on task : e.g. 20:00:00 -> 20 hours >
#SBATCH --output=<stdout filename : relative to directory of scheduling : e.g. stdout-%f.out>

module purge
module load python/3.10.2

python3 extract_image_caption.py -i <input directory> -o <output directory> -n <number of workers> -l <log level: info | debug> -flatten <flatten output directory or not, should use True>