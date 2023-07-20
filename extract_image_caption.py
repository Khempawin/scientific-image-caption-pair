import logging
import re
import io
import os
import xml.etree.ElementTree as ET
import tarfile
import argparse
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


def process_graphic(tar_archive: TarFile, node: Element, document_id: str, output_image_dir: str, logger: Logger=None):
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
    
    image_file = tar_archive.extractfile("{}/{}".format(document_id, image_name))
    
    # Save image to output directory
    with open(f"{output_image_dir}/{image_name}", "wb") as f:
        f.writelines(image_file.readlines())

    return image_name


def get_image_type(node: Element):
    if node.tag == "fig":
        return "figure"
    else:
        return "other"
    

def process_node_with_graphic(tar_archive: TarFile, node: Element, document_id: str, output_image_dir: str, logger: Logger=None):
    if logger is None:
        logger = logging
    record_dict = dict()
    record_dict["document_id"] = document_id
    record_dict["caption"] = process_caption(node)

    image_path = process_graphic(tar_archive, node, document_id=document_id, output_image_dir=output_image_dir, logger=logger)

    if(image_path is None):
        return None

    record_dict["image_path"] = image_path
    record_dict["image_type"] = get_image_type(node)
    return record_dict


def process_document_tar(entry: DirEntry, output_image_dir: str="output/images", logger: Logger=None):
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
                                                document_id=document_id, 
                                                output_image_dir=output_image_dir) for figure in figure_nodes]
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


def process_tar_dir(target_dir:str, output_dir:str, first_level_code:str, second_level_code:str, logger: Logger=None):
    if logger is None:
        logger = logging
    start_time = time()
    documents = os.scandir(target_dir)

    output_dir_suffix = f"{first_level_code}_{second_level_code}"
    output_dir_prefix = output_dir

    output_dir_base = Path(f"{output_dir_prefix}/output_{output_dir_suffix}")
    output_dir_base.mkdir(parents=True, exist_ok=True)
    output_image_dir = Path(f"{output_dir_prefix}/output_{output_dir_suffix}/images")
    output_image_dir.mkdir(parents=True, exist_ok=True)

    record_list = list()

    for doc in list(documents):
        subrecord_list = process_document_tar(doc,
                                              output_image_dir=output_image_dir)
        record_list.extend(subrecord_list)

    save_records_to_csv(output_dir_base / "captions.csv", record_list, sep="|")
    end_time = time()
    logger.info("  Time for completion of {}/{}: {:.2f} seconds".format(first_level_code, second_level_code, (end_time-start_time)))
    return record_list


def extract_all(main_dir: str, output_path: str="processed", n_workers=1, log_level=logging.INFO):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=output_dir / "log.out",
                        level=log_level,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    
    logger = logging.getLogger(__name__)

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
                             logger) for second_level_dir in second_level]


def parse_log_level(level: str):
    if(level == "debug"):
        return logging.DEBUG
    elif (level == "info"):
        return logging.INFO
    else:
        return logging.INFO


def main():
    parser = argparse.ArgumentParser(description="""
Image-Caption pair extraction from scientific papers
""")
    parser.add_argument("-i", "--input_dir", help="input directory containing directory of tar files", required=True)
    parser.add_argument("-o", "--output_dir", help="output directory", required=True)
    parser.add_argument("-n", "--n_workers", help="number of worker processes", default=1)
    parser.add_argument("-l", "--log_level", help="log level [0: debug, 1: info (normal operation)]", default="info")
    arg_list = parser.parse_args()
    main_dir = arg_list.input_dir
    output_path = arg_list.output_dir
    n_workers = int(arg_list.n_workers)
    log_level = parse_log_level(arg_list.log_level)
    extract_all(main_dir=main_dir,
                output_path=output_path,
                n_workers=n_workers,
                log_level=log_level)


if __name__ == "__main__":
    main()