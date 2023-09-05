from mpi4py import MPI
import os
import pandas as pd
import argparse
from typing import Dict, List, TypedDict
from pathlib import Path
import shutil
from transformers import AutoTokenizer

class ArgDict(TypedDict):
    input_dir: str
    output_dir: str
    image_output: bool


TOKENIZER = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def tokenize(text):
    output = TOKENIZER([text], return_tensors="pt")
    return output["input_ids"].shape[1]


def get_image_path(
        first_level_name: str,
        first_level_path: str,
        second_level: str,
        image_path: str
):
    return f"{first_level_path}/{first_level_name}_{second_level}_images/{image_path}"


def process_image_record(
        output_path: Path,
        first_level_name: str,
        first_level_path: str,
        second_level: str,
        image_path: str,
        include_image: bool=False
):
    src_path = Path(get_image_path(
        first_level_name=first_level_name,
        first_level_path=first_level_path,
        second_level=second_level,
        image_path=image_path
    ))

    if not src_path.is_file():
        return False

    if include_image:
        dest_dir = Path(f"{output_path}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src=src_path, dst=f"{dest_dir}/{image_path}")
    return True


def process_dir(dir: Dict, output_dir: str, image_output: bool=False):
    # Read parquet partition
    df = pd.read_parquet("{}/captions".format(dir["path"]))
    # Compute context validity by pandas apply on each record in partition
    df["fit_context"] = df.apply(lambda row: tokenize(row["caption"]), axis=1)
    # Prepare output directory
    output_path = Path(f"{output_dir}/captions")
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare image output directory
    suffix = dir["name"][-2:]
    output_image_path = Path(f"{output_dir}/image_{suffix}")
    if image_output:
        output_image_path.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    df["image_file_exist"] = df.apply(lambda row: process_image_record(
        output_path=output_image_path,
        first_level_name=suffix,
        first_level_path=dir["path"],
        second_level=row["second_level_dir"],
        image_path=row["image_path"],
        include_image=image_output
    ), axis=1)

    # Save result into parquet output dir
    df["first_level_dir"] = suffix
    df.to_parquet(f"{output_path}/{suffix}.parquet")
    return dir["name"]


def process_partition(dir_list: List[Dict], output_dir: str, image_output: bool=False):
    for dir in dir_list:
        process_dir(dir, output_dir, image_output)
    return


def partition_list(input_list, n_part):
    total = len(input_list)
    remainder = (len(input_list)) % n_part
    part_size = max(int((len(input_list))/n_part), 1)

    count = 0
    marker_list = list()
    for start in range(0, total, part_size):
        if(count >= n_part):
            break
        marker_list.append((start, start + part_size))
        count += 1

    if remainder == 0:
        return marker_list

    last_part = (marker_list[-1][0], None)
    return marker_list[:-1] + [last_part]


def send_partition(input_list, n_part, marker_list, comm):
    for i in range(0, n_part):
        if(i >= len(marker_list) or i == 0):
            continue
        start = marker_list[i][0]
        end = marker_list[i][1]
        active_part = input_list[start: end] if end else input_list[start:]
        # Send to partition
        comm.send(active_part, i, i)
        # print("{} | ({} -> {}) : {}".format(i, start, end, len(active_part)))
    return input_list[marker_list[0][0]: marker_list[0][1]]


def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    arg_dict = None
    if rank == 0:
        parser = argparse.ArgumentParser(description="""
            Compute context length with CLIP tokenizer of scientific image captions
        """)
        parser.add_argument(
            "-i", "--input_dir", help="input directory containing parquet files, accessible via network by every node", required=True)
        parser.add_argument(
            "-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
        parser.add_argument(
            "-img", "--image_out", help="copy image from input directory or not", default=False, required=False
        )
        arg_list = parser.parse_args()
        arg_dict = ArgDict(
            input_dir=arg_list.input_dir,
            output_dir=arg_list.output_dir,
            image_output=parse_boolean(arg_list.image_out)
        )

        # Send output dir to other processes
        arg_dict: ArgDict = comm.bcast(arg_dict, root=0)

        # List directories
        dir_list = [{
            "path": dir.path,
            "name": dir.name
        } for dir in os.scandir(arg_dict["input_dir"]) if dir.is_dir()]
        # Partition list of directories
        partition_boundaries = partition_list(dir_list, nprocs)
        # Send partition to other processes
        self_partition = send_partition(
            dir_list, nprocs, partition_boundaries, comm)

        # Process self partition
        process_partition(self_partition, arg_dict["output_dir"], arg_dict["image_output"])
    else:
        arg_dict = comm.bcast(arg_dict, root=0)
        dir_list = comm.recv(source=0, tag=rank)
        # Process dir_list
        process_partition(dir_list, arg_dict["output_dir"], arg_dict["image_output"])


if __name__ == "__main__":
    main()
