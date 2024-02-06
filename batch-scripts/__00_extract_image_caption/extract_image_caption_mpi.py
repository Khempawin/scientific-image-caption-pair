import logging
import re
import io
import os
import xml.etree.ElementTree as ET
import tarfile
import argparse
import pandas as pd
import concurrent.futures
from pathlib import Path
from time import time
from os import DirEntry
from xml.etree.ElementTree import Element
from typing import List, TypedDict, Any, Optional
from tarfile import TarFile
from logging import Logger
from mpi4py import MPI
from functools import reduce
from zipfile import ZipFile


class ArgDict(TypedDict):
    main_dir: str
    output_path: str
    flatten_output_directory: bool
    n_workers: int
    log_level: int
    omit_image_file: bool
    output_caption_file_type: str


class TarDir(TypedDict):
    dir_path: str
    first_level: str
    second_level: str


class GraphicDict(TypedDict):
    document_id: str
    caption: str
    image_path: Optional[str]
    image_type: str
    first_level_dir: str
    second_level_dir: str
    section: str
    journal_name: Optional[str]
    article_title: Optional[str]
    subjects: Optional[List[str]]
    authors: Optional[List[str]]


def extract_directory(target_root: str) -> List[DirEntry]:
    return [x for x in os.scandir(target_root) if len(x.name) == 2 and x.is_dir()]


def load_cleaned_xml_from_str(xml_string: str):
    cleaned_text = re.sub(r"<xref[^>]*>[^<]*<\/xref>", "", xml_string)
    cleaned_text = re.sub(
        r"(<bold[^>]*>|<\/bold>|<italic[^>]*>|<\/italic>|<sub( [^>]*|>)|<\/sub>)", "", cleaned_text)
    cleaned_text = re.sub(r"\&#x.{5};", "", cleaned_text)

    f = io.StringIO(cleaned_text)
    return ET.parse(f)


def node_has_graphic(node: Element):
    children_tag_set = set([child.tag for child in node.findall("*")])
    return "graphic" in children_tag_set


def process_caption(node: Element) -> str:
    # Find caption node
    caption_node = node.find("./caption")

    if(caption_node is None):
        return ""

    # For captions with <p>
    caption_p = caption_node.findtext("./p")
    caption_p = caption_p.strip().replace("\n", "") if caption_p else ""
    # For captions with <title>
    caption_title = caption_node.findtext("./title")
    caption_title = caption_title.strip().replace("\n", "") if caption_title else ""
    return "{}{}".format(caption_title, caption_p)


def process_graphic(
        tar_archive: TarFile, 
        node: Element, 
        first_level_code: str, 
        second_level_code: str, 
        document_id: str, 
        output_image_zip: ZipFile=None, 
        logger: Logger = None, 
        omit_image_file: bool = True):
    if logger is None:
        logger = logging
    IMAGE_FILE_EXTENSIONS = [".jpg", ".png", ".gif", ".tif"]
    graphic_node = node.find("./graphic")

    attribute_keys = [key for key in list(
        graphic_node.keys()) if key.endswith("href")]

    if(len(attribute_keys) == 0):
        logger.error("Error invalid reference to image")
        return None

    graphic_ref = graphic_node.get(attribute_keys[0])

    image_name = None

    if(re.search(r".png|.jpg|.gif|.tif[f]?$", graphic_ref)):
        image_name = graphic_ref
    else:
        # Try file extensions
        archive_files = tar_archive.getnames()
        for extension in IMAGE_FILE_EXTENSIONS:
            test_path = "{}/{}{}".format(document_id, graphic_ref, extension)
            if(test_path in archive_files):
                image_name = "{}{}".format(graphic_ref, extension)
                break

    if(image_name is None):
        return None

    saved_image_name = f"{first_level_code}_{second_level_code}_{document_id}_{image_name}"

    if(omit_image_file):
        return saved_image_name

    image_file = tar_archive.extractfile(
        "{}/{}".format(document_id, image_name))

    # Save image to output zip
    output_image_zip.writestr(saved_image_name, image_file.read())

    return saved_image_name


def get_image_type(node: Element):
    if node.tag == "fig":
        return "figure"
    else:
        return "other"


def process_node_with_graphic(tar_archive: TarFile,
                              node: Element,
                              first_level_code: str,
                              second_level_code: str,
                              document_id: str,
                              section: str,
                              journal_name: str=None,
                              article_title: str=None,
                              subjects: List[str]=None,
                              authors: List[str]=None,
                              output_image_zip: ZipFile=None,
                              logger: Logger = None,
                              omit_image_file: bool = True) -> GraphicDict:
    if logger is None:
        logger = logging
    logger.debug("Start node with graphic")
    record_dict = GraphicDict()
    record_dict["document_id"] = document_id
    record_dict["caption"] = process_caption(node)

    image_path = process_graphic(tar_archive,
                                 node,
                                 first_level_code=first_level_code,
                                 second_level_code=second_level_code,
                                 document_id=document_id,
                                 output_image_zip=output_image_zip,
                                 logger=logger,
                                 omit_image_file=omit_image_file)

    if(image_path is None):
        return None

    record_dict["image_path"] = image_path
    record_dict["image_type"] = get_image_type(node)
    record_dict["first_level_dir"] = first_level_code
    record_dict["second_level_dir"] = second_level_code
    record_dict["section"] = section
    record_dict["journal_name"] = journal_name
    record_dict["article_title"] = article_title
    record_dict["subjects"] = subjects
    record_dict["authors"] = authors
    return record_dict


def is_section_node(node: Element):
    return node.tag == "sec"


def get_section_title(section_node: Element):
    return section_node.find("title").text


def process_document_tar(entry: DirEntry,
                         first_level_code: str,
                         second_level_code: str,
                         output_image_zip: ZipFile = None,
                         logger: Logger = None,
                         omit_image_file: bool = True):
    if logger is None:
        logger = logging
    # Open tar file
    try:
        tar_archive = tarfile.open(entry.path, mode="r:gz")
        logger.debug(entry.path)
    except:
        logger.error("Failed to process : {}".format(entry.path))
        return []

    # Get nxml file name
    try:
        nxml_file_names = [
            file_name for file_name in tar_archive.getnames() if file_name.endswith(".nxml")]
    except:
        logger.error("Failed to decompress : {}".format(entry.path))
        tar_archive.close()
        return []

    # Check if nxml file exists
    if(len(nxml_file_names) == 0):
        tar_archive.close()
        return []
    nxml_file_name = nxml_file_names[0]
    # Get document id
    document_id = nxml_file_name.split("/")[0]
    logger.debug(nxml_file_name)
    logger.debug("Document id : {}".format(document_id))

    try:
        # Extract .nxml file to memory
        nxml_file = tar_archive.extractfile(nxml_file_name)
        nxml_content = "".join([line.decode("utf-8")
                                for line in nxml_file.readlines()])

        # Parse .nxml as cleaned tree
        tree = load_cleaned_xml_from_str(nxml_content)
        record_list = list()

        # Extract article metadata
        meta_data_node = [node for node in tree.iter() if node.tag == "article-meta" or node.tag == "journal-meta"]
        journal_meta_node = [node for node in meta_data_node if node.tag == "journal-meta"][0]
        article_meta_node = [node for node in meta_data_node if node.tag == "article-meta"][0]
        journal_meta_children = [node for node in journal_meta_node.iter()]
        article_meta_children = [node for node in article_meta_node.iter()]
        journal_name = [node.text for node in journal_meta_children if node.tag == "journal-title"]
        subjects = [node.text for node in article_meta_children if node.tag == "subject"]
        article_title = [node.text for node in article_meta_children if node.tag == "article-title"][0]
        
        authors = [node for node in article_meta_children if node.tag == "contrib"]
        authors = [name.find("name") for name in authors]
        authors = ["{} {}".format(name.find("given-names").text, name.find("surname").text) for name in authors]


        # Extract section nodes from tree
        body_node = tree.find("body")
        section_nodes = [node for node in body_node.findall("sec")]
        figure_nodes = list()
        for section_node in section_nodes:
            section_title = get_section_title(section_node)
            sub_figure_nodes = [{"node": node, "section": section_title} for node in section_node.iter() if node_has_graphic(node)]
            figure_nodes.extend(sub_figure_nodes)

        if len(figure_nodes) == 0:
            return []

        logger.debug("Got nodes")
        
        record_list = [process_node_with_graphic(tar_archive=tar_archive,
                                                 node=figure["node"],
                                                 first_level_code=first_level_code,
                                                 second_level_code=second_level_code,
                                                 document_id=document_id,
                                                 section=figure["section"],
                                                 journal_name=journal_name,
                                                 article_title=article_title,
                                                 subjects=subjects,
                                                 authors=authors,
                                                 output_image_zip=output_image_zip,
                                                 omit_image_file=omit_image_file) for figure in figure_nodes]
        logger.debug("\tDone")
        record_list = list(filter(lambda x: x is not None, record_list))

        # Close tar file
        tar_archive.close()

        # Return caption, document id
        return record_list
    except UnicodeDecodeError:
        logging.error(f"Unicode Decode Error on {document_id}")
        tar_archive.close()
        return []
    except:
        logger.error(f"Error parsing xml of {entry.path}", exc_info=True)
        # Close tar file
        tar_archive.close()
        # Return caption, document id
        return []


def process_tar_dir(target_dir: str,
                    output_dir: str,
                    first_level_code: str,
                    second_level_code: str,
                    flatten_output_dir=False,
                    omit_image_file: bool = True,
                    output_caption_file_type: str = "parquet"):
    logger = logging.getLogger(__name__)
    start_time = time()
    documents = os.scandir(target_dir)

    output_dir_suffix = f"{first_level_code}_{second_level_code}"
    output_dir_prefix = output_dir

    output_dir_base = Path(f"{output_dir_prefix}/output_{output_dir_suffix}") if flatten_output_dir else Path(
        f"{output_dir_prefix}/output_{first_level_code}")
    output_dir_base.mkdir(parents=True, exist_ok=True)
    output_caption_base = output_dir_base / "caption"
    output_caption_base.mkdir(parents=True, exist_ok=True)
        
    output_image_zip = None
    output_image_dir = Path(f"{output_dir_prefix}/output_{output_dir_suffix}/images") if flatten_output_dir else Path(
        f"{output_dir_prefix}/output_{first_level_code}/{first_level_code}_{second_level_code}_images")
    if(not omit_image_file):
        output_image_zip = ZipFile(f"{output_image_dir}.zip", "w")

    record_list = list()

    for doc in list(documents):
        if(not doc.is_file() or not doc.name.endswith("tar.gz")):
            continue
        subrecord_list = process_document_tar(doc,
                                              first_level_code=first_level_code,
                                              second_level_code=second_level_code,
                                              output_image_zip=output_image_zip,
                                              omit_image_file=omit_image_file)
        record_list.extend(subrecord_list)

    record_df = pd.DataFrame(record_list)
    if(output_image_zip is not None and not omit_image_file):
        output_image_zip.close()
    if(output_caption_file_type == "parquet"):
        parquet_path = output_caption_base / \
            "captions.parquet" if flatten_output_dir else output_caption_base / \
            f"{first_level_code}_{second_level_code}_captions.parquet"
        record_df.to_parquet(parquet_path, compression="gzip")
    else:
        csv_file_path = output_caption_base / "captions.csv" if flatten_output_dir else output_caption_base / \
            f"{first_level_code}_{second_level_code}_captions.csv"
        record_df.to_csv(csv_file_path, sep="|")

    end_time = time()
    logger.info("  Time for completion of {}/{}: {:.2f} seconds containing {}".format(
        first_level_code, second_level_code, (end_time-start_time), len(record_list)))
    return record_list


def configure_logging(process_rank: int, output_path: str, log_level: int):
    # Initialize output path
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize log file
    logging.basicConfig(filename=output_dir / f"log.{process_rank}.out",
                        level=log_level,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    if (process_rank == 0):
        logging.info("Start process")
    else:
        logging.info(f"Initialized process {process_rank}")
    return


def get_all_tar_dir(main_dir: str) -> List[TarDir]:
    # get first level dir
    first_level = [(dir, dir.name)
                   for dir in sorted(extract_directory(main_dir), key=str)]
    # for each first level dir get second level dir
    second_level = list(map(lambda parent_dir: [TarDir(dir_path=dir.path, first_level=parent_dir[1],
                        second_level=dir.name) for dir in extract_directory(parent_dir[0].path)], first_level))
    # flatten dir
    flatten_dir_list = reduce(lambda a, b: a+b, second_level)
    return sorted(flatten_dir_list, key=str)


def partition_sequence(input_list: List[Any], n_partition: int):
    total = len(input_list)
    base_partition_size = int(total / n_partition)
    partition_size_list = [base_partition_size] * n_partition
    remainder = total % n_partition
    # divide remainder amoung partitions
    index = 0
    while(remainder > 0):
        partition_size_list[index] += 1
        remainder -= 1
        index += 1

    partition_list = list()
    start_index = 0
    for size in partition_size_list:
        partition_list.append(input_list[start_index: start_index+size])
        start_index += size

    # return list of list(partition)
    return partition_list


def parse_log_level(level: str):
    if(level == "debug"):
        return logging.DEBUG
    elif (level == "info"):
        return logging.INFO
    else:
        return logging.INFO


def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    arg_dict = None
    dir_list = None

    if rank == 0:
        parser = argparse.ArgumentParser(description="""
    Image-Caption pair extraction from scientific papers
    """)
        parser.add_argument(
            "-i", "--input_dir", help="input directory containing directory of tar files", required=True)
        parser.add_argument("-o", "--output_dir",
                            help="output directory", required=True)
        parser.add_argument("-flatten", "--flatten_second_level",
                            help="flatten second level output directory", default=False)
        parser.add_argument("-n", "--n_workers",
                            help="number of workers per mpi process", default=1)
        parser.add_argument(
            "-l", "--log_level", help="log level [0: debug, 1: info (normal operation)]", default="info")
        parser.add_argument(
            "--omit_image_file", help="Omit image files [True | False]", default=False)
        parser.add_argument("--caption_output_type",
                            help="Specify output caption file format [csv | parquet]", default="parquet")
        arg_list = parser.parse_args()
        main_dir = arg_list.input_dir
        output_path = arg_list.output_dir
        flatten_output_directory = parse_boolean(arg_list.flatten_second_level)
        n_workers = int(arg_list.n_workers)
        log_level = parse_log_level(arg_list.log_level)
        omit_image_file = parse_boolean(arg_list.omit_image_file)
        output_caption_file_type = arg_list.caption_output_type

        # Package arguments as Argument Dict
        arg_dict = ArgDict(
            main_dir=main_dir,
            output_path=output_path,
            flatten_output_directory=flatten_output_directory,
            n_workers=n_workers,
            log_level=log_level,
            omit_image_file=omit_image_file,
            output_caption_file_type=output_caption_file_type
        )
        # Broadcast argument dict to each process
        arg_dict: ArgDict = comm.bcast(arg_dict, root=0)

        # Configure logging
        configure_logging(
            process_rank=rank,
            output_path=arg_dict["output_path"],
            log_level=arg_dict["log_level"]
        )

        # List all tar directories as list
        tar_dirs = get_all_tar_dir(main_dir=main_dir)

        # Partition tar dir list
        partitions = partition_sequence(
            input_list=tar_dirs,
            n_partition=nprocs
        )

        # Send partitions to other processes
        for i, partition in enumerate(partitions[:-1]):
            comm.send(partition, i+1, tag=i+1)

        dir_list = partitions[-1]
    else:
        arg_dict: ArgDict = comm.bcast(arg_dict, root=0)
        # Configure logging
        configure_logging(
            process_rank=rank,
            output_path=arg_dict["output_path"],
            log_level=arg_dict["log_level"]
        )

        # Receive partition to process
        dir_list: List[TarDir] = comm.recv(dir_list, source=0, tag=rank)

    # Process dir list
    with concurrent.futures.ProcessPoolExecutor(max_workers=arg_dict["n_workers"]) as executor:
        [executor.submit(
            process_tar_dir,
            dir["dir_path"],
            arg_dict["output_path"],
            dir["first_level"],
            dir["second_level"],
            arg_dict["flatten_output_directory"],
            arg_dict["omit_image_file"],
            arg_dict["output_caption_file_type"]
        )for dir in dir_list]
    logging.info(f"End of process {rank}")


if __name__ == "__main__":
    main()

