# Overview
This script is designed to extract image-caption pairs from a directory containing tar files of scientific articles.

# Instructions
## 1. Running on local machine
python3 extract_image_caption.py -i <input_directory> -o <output_directory> -n <number of workers> -l <log level: info | debug> -flatten <flatten output directory or not, should use True> --omit_image_file <Use to set to output image files or not [True | False]> --caption_output_type parquet <Caption file output type [csv | parquet]>

## 2. Submitting to cluster via slurm
### 2.1 Edit the script file "run_script.sh"
### 2.2 Copy "run_script.sh" and "extract_image_caption.py" to front facing node
### 2.3 execute command "sbatch run_script.sh"


# Installing and Preparation
## Conda (preferred)
- Create virtual environment  
<code>conda create -p `<path to directory>` python=3.11 </code>
- Activate virtual environment  
<code>conda activate `<path to directory`></code>

- Install Pytorch (https://pytorch.org/get-started/locally/)  
<code>conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia</code>
- Install pandas, dask, pyarrow  
<code>conda install pandas dask pyarrow</code>
- Install mpi4py (optional, use in batch script only)  
<code>conda install mpi4py</code>
- Install clip package  or  clip huggingface  
<code>conda install -c huggingface transformers</code>
- Install faiss package for nearest neighbor search   
<code>conda install -c conda-forge faiss</code>   
- Install other dependencies  
<code>conda install scikit-learn accelerate -c conda-forge</code>
- Install ScientificImageCaption Package  
<code>pip install git+https://github.com/Khempawin/scientific-image-caption-pair.git</code>  

- Dev interactive  
<code>conda install tqdm jupyter ipywidgets ipython</code>

## Python venv (Incomplete)
- Create virtual environment  
<code>python -m venv __directory__ </code>
- Activate virtual environment  
<code>. /path/to/virutal_env/bin/activate</code>
- Upgrade pip  
<code>python -m pip install --upgrade pip</code>

- Install Pytorch (https://pytorch.org/get-started/locally/)  
<code>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</code>
- Install pandas, dask, pyarrow  
<code>pip install pandas dask pyarrow</code>
- Install clip package  or  clip huggingface
- Install ScientificImageCaption Package  


- Install mpi4py (optional, use in batch script only)

# Data pipeline

# Using batch scripts