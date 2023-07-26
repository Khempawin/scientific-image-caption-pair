# Overview
This script is designed to extract image-caption pairs from a directory containing tar files of scientific articles.

# Instructions
## 1. Running on local machine
python3 extract_image_caption.py -i <input_directory> -o <output_directory> -n <number of workers> -l <log level: info | debug> -flatten <flatten output directory or not, should use True> --omit_image_file <Use to set to output image files or not [True | False]>

## 2. Submitting to cluster via slurm
### 2.1 Edit the script file "run_script.sh"
### 2.2 Copy "run_script.sh" and "extract_image_caption.py" to front facing node
### 2.3 execute command "sbatch run_script.sh"