import logging
import re
import io
import os
import xml.etree.ElementTree as ET
import tarfile
import argparse
import pandas as pd
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor
from os import DirEntry
from xml.etree.ElementTree import Element
from typing import List, Dict, Any
from tarfile import TarFile
from logging import Logger


def extract_directory(target_root:str) -> List[DirEntry]:
    return [x for x in os.scandir(target_root) if len(x.name) == 2 and x.is_dir()]


def save_records_to_csv(file_path: str, record_list: List[Dict[str, Any]], sep="|"):
    if(len(record_list) == 0):
        return
    with open(file_path, "w") as f:
        columns = ["id"]
        columns.extend(list(record_list[0].keys()))
        # Write headers
        f.write(sep.join(columns) + "\n")
        for i, record in enumerate(record_list):
            values = [str(i)]
            for k in columns[1:]:
                values.append(record[k])
            f.write(sep.join(values) + "\n")
    return


def load_cleaned_xml_from_str(xml_string: str):
    cleaned_text = re.sub(r"<xref[^>]*>[^<]*<\/xref>", "", xml_string)
    cleaned_text = re.sub(r"(<bold[^>]*>|<\/bold>|<italic[^>]*>|<\/italic>|<sub( [^>]*|>)|<\/sub>)", "", cleaned_text)
    cleaned_text = re.sub(r"\&#x.{5};", "", cleaned_text)

    f = io.StringIO(cleaned_text)
    return ET.parse(f)


def node_has_graphic(node: Element):
    children_tag_set = set([child.tag for child in node.findall("*")])
    return "graphic" in children_tag_set


def process_caption(node: Element):
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


def process_graphic(tar_archive: TarFile, node: Element, first_level_code: str, second_level_code: str, document_id: str, output_image_dir: str, logger: Logger=None, omit_image_file: bool=True):
    if logger is None:
        logger = logging
    IMAGE_FILE_EXTENSIONS = [".jpg", ".png", ".gif", ".tif"]
    graphic_node = node.find("./graphic")

    attribute_keys = [key for key in list(graphic_node.keys()) if key.endswith("href")]

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

    image_file = tar_archive.extractfile("{}/{}".format(document_id, image_name))
    
    # Save image to output directory
    with open(f"{output_image_dir}/{saved_image_name}", "wb") as f:
        f.writelines(image_file.readlines())

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
                              output_image_dir: str, 
                              logger: Logger=None, 
                              omit_image_file:bool=True):
    if logger is None:
        logger = logging
    record_dict = dict()
    record_dict["document_id"] = document_id
    record_dict["caption"] = process_caption(node)

    image_path = process_graphic(tar_archive, 
                                 node, 
                                 first_level_code=first_level_code,
                                 second_level_code=second_level_code,
                                 document_id=document_id, 
                                 output_image_dir=output_image_dir, 
                                 logger=logger,
                                 omit_image_file=omit_image_file)

    if(image_path is None):
        return None

    record_dict["image_path"] = image_path
    record_dict["image_type"] = get_image_type(node)
    record_dict["first_level_dir"] = first_level_code
    record_dict["second_level_dir"] = second_level_code
    return record_dict


def process_document_tar(entry: DirEntry, 
                         first_level_code:str, 
                         second_level_code:str, 
                         output_image_dir: str="output/images", 
                         logger: Logger=None, 
                         omit_image_file:bool=True):
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
        nxml_file_names = [file_name for file_name in tar_archive.getnames() if file_name.endswith(".nxml")]
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

    # Extract .nxml file to memory
    nxml_file = tar_archive.extractfile(nxml_file_name)
    nxml_content = "".join([line.decode("utf-8") for line in nxml_file.readlines()])

    # Parse .nxml as cleaned tree
    try:
        tree = load_cleaned_xml_from_str(nxml_content)
        record_list = list()

        # Extract image(figure) file names and captions from tree
        figure_nodes = [node for node in tree.iter() if node_has_graphic(node)]

        if len(figure_nodes) == 0:
            return []

        record_list = [process_node_with_graphic(tar_archive=tar_archive,
                                                node=figure, 
                                                first_level_code=first_level_code,
                                                second_level_code=second_level_code,
                                                document_id=document_id, 
                                                output_image_dir=output_image_dir,
                                                omit_image_file=omit_image_file) for figure in figure_nodes]
        record_list = list(filter(lambda x: x is not None, record_list))
        
        # Close tar file
        tar_archive.close()

        # Return caption, document id
        return record_list
    except:
        logger.error(f"Error parsing xml of {entry.path}")
        # Close tar file
        tar_archive.close()
        # Return caption, document id
        return []


def process_tar_dir(target_dir:str, 
                    output_dir:str, 
                    first_level_code:str, 
                    second_level_code:str, 
                    flatten_output_dir=False, 
                    omit_image_file:bool=True,
                    output_caption_file_type: str="csv"):
    logger = logging.getLogger(__name__)
    start_time = time()
    documents = os.scandir(target_dir)

    output_dir_suffix = f"{first_level_code}_{second_level_code}"
    output_dir_prefix = output_dir

    if(flatten_output_dir):
        output_dir_base = Path(f"{output_dir_prefix}/output_{first_level_code}")
        output_dir_base.mkdir(parents=True, exist_ok=True)
        output_image_dir = Path(f"{output_dir_prefix}/output_{first_level_code}/{first_level_code}_{second_level_code}_images")
        output_image_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_base = Path(f"{output_dir_prefix}/output_{output_dir_suffix}")
        output_dir_base.mkdir(parents=True, exist_ok=True)
        output_image_dir = Path(f"{output_dir_prefix}/output_{output_dir_suffix}/images")
        output_image_dir.mkdir(parents=True, exist_ok=True)

    record_list = list()

    for doc in list(documents):
        subrecord_list = process_document_tar(doc,
                                              first_level_code=first_level_code,
                                              second_level_code=second_level_code,
                                              output_image_dir=output_image_dir,
                                              omit_image_file=omit_image_file)
        record_list.extend(subrecord_list)

    if(output_caption_file_type == "parquet"):
        record_df = pd.DataFrame(record_list)
        parquet_path = output_dir_base / f"{first_level_code}_{second_level_code}_captions.parquet.gzip" if flatten_output_dir else output_dir_base / "captions.parquet.gzip"
        record_df.to_parquet(parquet_path, compression="gzip")
    else:
        csv_file_path = output_dir_base / f"{first_level_code}_{second_level_code}_captions.csv" if flatten_output_dir else output_dir_base / "captions.csv"

        save_records_to_csv(csv_file_path, 
                            record_list, 
                            sep="|")
    end_time = time()
    logger.info("  Time for completion of {}/{}: {:.2f} seconds containing {}".format(first_level_code, second_level_code, (end_time-start_time), len(record_list)))
    return record_list
  
  
def extract_all(main_dir: str, 
                output_path: str="processed", 
                n_workers=1, 
                log_level=logging.INFO, 
                flatten_output_dir=False, 
                omit_image_file:bool=True,
                output_caption_file_type: str="csv"):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=output_dir / "log.out",
                        level=log_level,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    logging.info("Start process")
    first_level = sorted(extract_directory(main_dir), key=str)

    for first_level_dir in first_level:
        logging.info(f"Processing first level directory : {first_level_dir.path}")
        second_level = sorted(extract_directory(first_level_dir.path), key=str)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            [executor.submit(process_tar_dir,
                             second_level_dir.path,
                             output_dir.resolve(),
                             first_level_dir.name,
                             second_level_dir.name,
                             flatten_output_dir,
                             omit_image_file,
                             output_caption_file_type) for second_level_dir in second_level]


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
    parser = argparse.ArgumentParser(description="""
Image-Caption pair extraction from scientific papers
""")
    parser.add_argument("-i", "--input_dir", help="input directory containing directory of tar files", required=True)
    parser.add_argument("-o", "--output_dir", help="output directory", required=True)
    parser.add_argument("-flatten", "--flatten_second_level", help="flatten second level output directory", default=False)
    parser.add_argument("-n", "--n_workers", help="number of worker processes", default=1)
    parser.add_argument("-l", "--log_level", help="log level [0: debug, 1: info (normal operation)]", default="info")
    parser.add_argument("--omit_image_file", help="Omit image files [True | False]", default=False)
    parser.add_argument("--caption_output_type", help="Specify output caption file format [csv | parquet]", default="csv")
    arg_list = parser.parse_args()
    main_dir = arg_list.input_dir
    output_path = arg_list.output_dir
    flatten_output_directory = parse_boolean(arg_list.flatten_second_level)
    n_workers = int(arg_list.n_workers)
    log_level = parse_log_level(arg_list.log_level)
    omit_image_file = parse_boolean(arg_list.omit_image_file)
    output_caption_file_type = arg_list.caption_output_type
    extract_all(main_dir=main_dir,
                output_path=output_path,
                n_workers=n_workers,
                log_level=log_level,
                flatten_output_dir=flatten_output_directory,
                omit_image_file=omit_image_file,
                output_caption_file_type=output_caption_file_type)


if __name__ == "__main__":
    main()