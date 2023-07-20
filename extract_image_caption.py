import logging
import re
import io
import os
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from time import time
from os import DirEntry
from xml.etree.ElementTree import Element
from typing import List, Dict, Any


def extract_directory(target_root:str) -> List[DirEntry]:
    return [x for x in os.scandir(target_root) if len(x.name) == 2 and x.is_dir()]


def save_records_to_csv(file_path: str, record_list: List[Dict[str, Any]], sep="|"):
    if(len(record_list) == 0):
        return
    with open(file_path, "w") as f:
        columns = ["id"]
        columns.extend(list(record_list[0].keys()))
        # Write headers
        f.write("|".join(columns) + "\n")
        for i, record in enumerate(record_list):
            values = [str(i)]
            for k in columns[1:]:
                values.append(record[k])
            f.write("|".join(values) + "\n")
    return


def load_cleaned_xml(file_path: str):
    with open(file_path) as f:
        raw_text = "".join(f.readlines())

    cleaned_text = re.sub(r"<xref[^>]*>[^<]*<\/xref>", "", raw_text)
    cleaned_text = re.sub(r"(<bold>|<\/bold>|<italic>|<\/italic>)", "", cleaned_text)
    cleaned_text = re.sub(r"\&#x.{5};", "", cleaned_text)

    f = io.StringIO(cleaned_text)
    return ET.parse(f)

def node_has_graphic(node: Element):
    children_tag_set = set([child.tag for child in node.findall("*")])
    return "graphic" in children_tag_set


def process_caption(node: Element):
    # For captions with <p>
    caption_p = node.find("./caption").findtext("./p")
    caption_p = caption_p.strip().replace("\n", "") if caption_p else ""
    # For captions with <title>
    caption_title = node.find("./caption").findtext("./title")
    caption_title = caption_title.strip().replace("\n", "") if caption_title else ""
    return "{}{}".format(caption_title, caption_p)


def process_graphic(node: Element, document_id: str, output_image_dir: str):
    graphic_node = node.find("./graphic")

    attribute_keys = [key for key in list(graphic_node.keys()) if key.endswith("href")]

    if(len(attribute_keys) == 0):
        print("Error invalid reference to image")
        return None
    
    image_path = "{}.jpg".format(graphic_node.get(attribute_keys[0]))
    
    # Move image to output directory
    logging.debug("Moving image with command : mv {}/{} {}/".format(document_id, image_path, output_image_dir))
    subprocess.run("mv {}/{} {}/".format(document_id, image_path, output_image_dir), shell=True)
    return image_path


def process_node_with_graphic(node: Element, document_id: str, output_image_dir: str):
    record_dict = dict()
    record_dict["document_id"] = document_id
    record_dict["caption"] = process_caption(node)

    image_path = process_graphic(node, document_id=document_id, output_image_dir=output_image_dir)

    if(image_path is None):
        return None

    record_dict["image_path"] = image_path
    return record_dict


def process_document_tar(entry: DirEntry, output_image_dir: str="output/images", remove_dir=True):
    # Extract nxml file
    return_code = subprocess.run("tar -zxf {} --wildcards *.nxml".format(entry.path), shell=True)
    if return_code.returncode != 0:
        raise Exception("Error processing document at : {}".format(entry.path))
    document_id = entry.name.split(".")[0]

    metadata_file_name = os.listdir(document_id)[0]
    metadata_file_path = "{}/{}".format(document_id, metadata_file_name)

    # Extract image(figure) file names from nxml + captions
    tree = load_cleaned_xml(metadata_file_path)

    figure_nodes = [node for node in tree.iter() if node_has_graphic(node)]

    if len(figure_nodes) == 0:
        # Remove temp directory
        if(remove_dir):
            shutil.rmtree("{}".format(document_id))
        return []

    logging.debug("Extracting image with command : tar -zxf {} --wildcards *.jpg".format(entry.path))
    res = subprocess.run("tar -zxf {} --wildcards *.jpg".format(entry.path), shell=True)
    if res.returncode != 0:
        print("Error extracting images document id : {}".format(document_id))

    record_list = [process_node_with_graphic(figure, document_id=document_id, output_image_dir=output_image_dir) for figure in figure_nodes]
    record_list = list(filter(lambda x: x is not None, record_list))

    # Remove temp directory
    if(remove_dir):
        shutil.rmtree("{}".format(document_id))
    # Return caption, document id
    return record_list


def process_tar_dir(target_dir:str, output_dir:str, first_level_code:str, second_level_code:str, remove_dir=True):
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
                                              output_image_dir=output_image_dir, remove_dir=remove_dir)
        record_list.extend(subrecord_list)

    save_records_to_csv(output_dir_base / "captions.csv", record_list, sep="|")
    return record_list
    
def extract_all(main_dir: str, output_path: str="processed", remove_temp_dir=True):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=output_dir / "log.out",
                        level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Start process")
    first_level = extract_directory(main_dir)

    for first_level_dir in first_level:
        logging.info(f"Processing first level directory : {first_level_dir.path}")
        for second_level_dir in sorted(extract_directory(first_level_dir.path)):
            logging.info(f"  Processing second level directory : {second_level_dir.path}")
            start_time = time()
            process_tar_dir(second_level_dir.path,
                            output_dir=output_dir.resolve(), 
                            first_level_code=first_level_dir.name,
                            second_level_code=second_level_dir.name,
                            remove_dir=remove_temp_dir)
            end_time = time()
            logging.info("  Time for completion of {}/{}: {:.2f} seconds".format(first_level_dir.name, second_level_dir.name, (end_time-start_time)))


def main():
    main_dir = "<base_2_level_directory_of_tar_files>"
    output_path = "processed"
    extract_all(main_dir=main_dir,
                output_path=output_path,
                remove_temp_dir=False)
    

if __name__ == "__main__":
    main()