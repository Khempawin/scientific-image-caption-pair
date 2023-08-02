from mpi4py import MPI
import os
import pandas as pd
import clip
import argparse
from typing import Dict, List


def tokenize(text):
    context_length = 77
    while(True):
        try:
            clip.tokenize(text, context_length=context_length)
            return context_length
        except:
            context_length += 8


def process_dir(dir: Dict, output_dir: str):
    # Read parquet partition
    df = pd.read_parquet(dir["path"])
    # Compute context validity by pandas apply on each record in partition
    df["fit_context"] = df.apply(lambda row: tokenize(row["caption"]), axis=1)
    # Save result into parquet output dir
    suffix = dir["name"][-2:]
    df["first_level_dir"] = suffix
    df.to_parquet(f"{output_dir}/{suffix}.parquet")
    return dir["name"]


def process_partition(dir_list: List[Dict], output_dir: str):
    for dir in dir_list:
        process_dir(dir, output_dir)
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
    return  marker_list[:-1] + [last_part]


def send_partition(input_list, n_part, marker_list, comm):
    for i in range(0, n_part):
        if(i >= len(marker_list) or i == 0):
            continue
        start = marker_list[i][0]
        end = marker_list[i][1]
        active_part = input_list[start: end] if end else input_list[start:]
        # Send to partition
        comm.send(active_part, i, i)
        print("{} | ({} -> {}) : {}".format(i, start, end, len(active_part)))
    return input_list[marker_list[0][0]: marker_list[0][1]]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    output_dir = None
    if rank == 0:
        parser = argparse.ArgumentParser(description="""
            Compute context length with CLIP tokenizer of scientific image captions
        """)
        parser.add_argument("-i", "--input_dir", help="input directory containing parquet files, accessible via network by every node", required=True)
        parser.add_argument("-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
        arg_list = parser.parse_args()
        input_dir = arg_list.input_dir
        output_dir = arg_list.output_dir

        # Send output dir to other processes
        output_dir = comm.bcast(output_dir, root=0)

        # List directories
        dir_list = [{
            "path" : dir.path,
            "name" : dir.name
        } for dir in os.scandir(input_dir)]
        # Partition list of directories
        partition_boundaries = partition_list(dir_list, nprocs)
        # Send partition to other processes
        self_partition = send_partition(dir_list, nprocs, partition_boundaries, comm)

        # Process self partition
        process_partition(self_partition, output_dir)
    else:
        output_dir = comm.bcast(output_dir, root=0)
        dir_list = comm.recv(source=0, tag=rank)
        # Process dir_list
        process_partition(dir_list, output_dir)


if __name__ == "__main__":
    main()