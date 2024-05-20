import pandas as pd
import io
import os
from typing import TypedDict, List
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from pathlib import Path
import logging


class DataSplit(TypedDict):
    name: str
    manifest: pd.DataFrame
    image_list: List[Image.Image]


def load_image_list_from_zipfile(zipfile_path: str, filename_list: List[str], output_zip: ZipFile) -> List[Image.Image]:
    image_count = 0
    with ZipFile(zipfile_path, "r") as inzip:
        available_filename_list = set(inzip.namelist())
        for filename in filename_list:
            if(filename in available_filename_list):
                try:
                    img_file = inzip.read(filename)
                    img = Image.open(io.BytesIO(img_file)).convert("RGB")
                    output_bytes = io.BytesIO()
                    img.save(output_bytes, format="jpeg")
                    output_zip.writestr(filename, output_bytes.getvalue())
                    image_count += 1
                except:
                    logging.error(f"Error with file : {filename}")
    return image_count


def get_image_list(input_split: DataSplit, input_img_dir: str, output_dir: Path) -> DataSplit:
    input_split["image_list"] = list()

    name = input_split.get("name")
    data = input_split.get("manifest")
    data = data.rename(columns={"image_path":"original_image_path"})
    data["image_path"] = data.apply(lambda row: f'{".".join(row["original_image_path"].split(".")[:-1])}.jpg', axis=1)
    # Update manifest on image names
    input_split["manifest"] = data

    img_list = list()

    # Get unique first_level_dir
    unique_first_level_dir = data["first_level_dir"].unique()

    logging.info(unique_first_level_dir)
    with ZipFile("{}/{}.zip".format(output_dir, name), mode="w") as output_zip:
        for first_level_dir in unique_first_level_dir:
            print("Processing {} : {}".format(name, first_level_dir))
            subdata = data[data["first_level_dir"] == first_level_dir]
            logging.info(f"{first_level_dir} : {subdata.shape}")
            loaded_img_count = load_image_list_from_zipfile(f'{input_img_dir}/{first_level_dir}_image.zip', list(subdata["original_image_path"]), output_zip)
            logging.info(f"\tLoaded {loaded_img_count} images")

    input_split["image_list"] = img_list

    data.to_parquet(f"{output_dir}/{name}.parquet", engine="pyarrow")


    return input_split

def main():
    output_dir = Path("/home/horton/datasets/meta-scir/dataset-plain")

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=output_dir / "log.out", level=logging.INFO, format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    input_img_dir = "/home/horton/datasets/meta-scir/images"
    
    df = pd.read_parquet("/home/horton/datasets/meta-scir/captions", engine="pyarrow").drop_duplicates(["document_id", "image_path", "caption"])
    df = df.reset_index().drop(columns=["index"])
    # df = df[(df["first_level_dir"] == "00" )]

    logging.info("SELECT DATA DONE")

    # Split Train Val Test
    train_data, val_data = train_test_split(df, test_size=0.2)

    val_data, test_data = train_test_split(val_data, test_size=0.5)

    data_list:List[DataSplit] = [
            {"name": "train", "manifest": train_data},
            {"name": "val", "manifest": val_data},
            {"name": "test", "manifest": test_data}
            ]

    logging.info("SPLIT TRAIN_VAL_TEST DONE")
    

    logging.info("LOADING IMAGES")

    data_with_img_list:List[DataSplit] = [get_image_list(data_split, input_img_dir, output_dir) for data_split in data_list]

    logging.info("IMAGE LOADING DONE")
    logging.info(len(data_with_img_list))
    logging.info("********************COMPLETED**********************")

if __name__ == "__main__":
    main()

