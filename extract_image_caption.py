import os
import subprocess
from os import DirEntry
from typing import List, Dict, Any
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from time import time
import logging
import re
import io

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

def process_tar_dir(target_dir:str, output_dir:str, first_level_code:str, second_level_code:str):
    documents = os.scandir(target_dir)

    output_dir_suffix = f"{first_level_code}_{second_level_code}"
    output_dir_prefix = output_dir

    output_dir_base = Path(f"{output_dir_prefix}/output_{output_dir_suffix}")
    output_dir_base.mkdir(parents=True, exist_ok=True)
    output_image_dir = Path(f"{output_dir_prefix}/output_{output_dir_suffix}/images")
    output_image_dir.mkdir(parents=True, exist_ok=True)

    record_list = list()

    for doc in list(documents)[:1]:
        subrecord_list = process_document_tar(doc,
                                              output_image_dir=output_image_dir, remove_dir=False)
        record_list.extend(subrecord_list)

    save_records_to_csv(output_dir_base / "captions.csv", record_list, sep="|")
    return record_list
    
def load_cleaned_xml(file_path: str):
    with open(file_path) as f:
        raw_text = "".join(f.readlines())

    cleaned_text = re.sub(r"<xref[^>]*>[^<]*<\/xref>", "", raw_text)
    cleaned_text = re.sub(r"(<bold>|<\/bold>|<italic>|<\/italic>)", "", cleaned_text)
    cleaned_text = re.sub(r"\&#x.{5};", "", cleaned_text)

    f = io.StringIO(cleaned_text)
    tree = ET.parse(f)
    return tree


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

    figure_nodes = [node for node in tree.iter() if node.tag == "fig"]

    if len(figure_nodes) == 0:
        # Remove temp directory
        if(remove_dir):
            shutil.rmtree("{}".format(document_id))
        return []

    logging.debug("Extracting image with command : tar -zxf {} --wildcards *.jpg".format(entry.path))
    res = subprocess.run("tar -zxf {} --wildcards *.jpg".format(entry.path), shell=True)
    if res.returncode != 0:
        print("Error extracting images document id : {}".format(document_id))

    record_list = list()

    for figure in figure_nodes:
        temp_record_dict = dict()
        temp_record_dict["document_id"] = document_id
        temp_caption_p = figure.find("./caption").findtext("./p")
        temp_caption_p = temp_caption_p.strip().replace("\n", "") if temp_caption_p else ""
        temp_caption_title = figure.find("./caption").findtext("./title")
        temp_caption_title = temp_caption_title.strip().replace("\n", "") if temp_caption_title else ""

        temp_record_dict["caption"] = temp_caption_title + temp_caption_p

        graphic_node = figure.find("./graphic")

        attribute_keys = [key for key in list(graphic_node.keys()) if key.endswith("href")]

        if(len(attribute_keys) == 0):
            print("Error invalid reference to image")
            continue
        
        temp_record_dict["image_path"] = "{}.jpg".format(graphic_node.get(attribute_keys[0]))
        # Move image to output directory
        logging.debug("Moving image with command : mv {}/{} {}/".format(document_id, temp_record_dict["image_path"], output_image_dir))
        subprocess.run("mv {}/{} {}/".format(document_id, temp_record_dict["image_path"], output_image_dir), shell=True)
        record_list.append(temp_record_dict)

    # Remove temp directory
    if(remove_dir):
        shutil.rmtree("{}".format(document_id))
    # Return caption, document id
    return record_list

def extract_all(main_dir: str, output_path: str="processed"):
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
                            second_level_code=second_level_dir.name)
            end_time = time()
            logging.info("  Time for completion of {}/{}: {:.2f} seconds".format(first_level_dir.name, second_level_dir.name, (end_time-start_time)))


def main():
    main_dir = "<base_2_level_directory_of_tar_files>"
    output_path = "processed"
    extract_all(main_dir=main_dir,
                output_path=output_path)
    

if __name__ == "__main__":
    main()