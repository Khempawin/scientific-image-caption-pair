import pandas as pd
import argparse
import shutil
from pathlib import Path
from transformers import (
    VisionTextDualEncoderProcessor,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
)
from PIL import Image
from typing import Tuple
import numpy as np
import torch
import time

# Used to enable VisionTextDualModel encoding of Text
DUMMY_IMAGE = Image.fromarray(np.random.randint(255, size=(224,224,3),dtype=np.uint8))

DUMMY_TEXT = "sample"


def get_full_image_path(row, image_root_dir: Path):
    return image_root_dir / f"image_{row['first_level_dir']}" / row["image_path"]


def load_image(image_path:str) -> Tuple[bool, Image.Image]:
    try:
        image = Image.open(image_path).convert("RGB")
        return True, image
    except:
        return False, DUMMY_IMAGE


def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def encode_section(
        records: pd.DataFrame, 
        first_level_dir: str, 
        model: AutoModel,
        processor: VisionTextDualEncoderProcessor,
        batch_size: int,
        device: str,
        image_root_dir: str,
        output_path: str
    ):
    DATA_SIZE = records.shape[0]

    # Divide records into batches
    batches = [list(range(i, i+batch_size)) if i+batch_size < DATA_SIZE else list(range(i, DATA_SIZE)) for i in range(0, DATA_SIZE, batch_size)]

    # Initialize encoded lists
    encoded_caption_list = list()
    encoded_image_list = list()
    load_status_list = list()

    # Process each batch
    for batch in batches:
        image_list = list()
        # Load images
        for i in batch:
            success, image = load_image(get_full_image_path(records.iloc[i], image_root_dir))
            load_status_list.append(success)
            image_list.append(image)
        # Select captions for batch
        caption_list = list(records.iloc[batch[0]:batch[-1]+1]["caption"])
        # Encode image and caption
        with torch.no_grad():
            inputs = processor(text=caption_list, images=image_list, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            outputs = model(**inputs)
        
        # Add encoded batch result to encoded lists
        encoded_caption_list.extend(list(np.asarray(outputs.text_embeds.to("cpu"))))
        encoded_image_list.extend(list(np.asarray(outputs.image_embeds.to("cpu"))))

    # Assign encoded results to original dataframe
    records["encoded_caption"] = encoded_caption_list
    records["encoded_image"] = encoded_image_list
    records["load_status"] = load_status_list

    # Save dataframe to parquet file
    valid_records = records[records["load_status"] == True]
    print("Processed records of {} : {} | Valid records : {}".format(first_level_dir, records.shape[0], valid_records.shape[0]))
    valid_records.to_parquet(output_path / f"encoded_caption_{first_level_dir}.parquet", engine="pyarrow")
    return

def main():
    parser = argparse.ArgumentParser(description="""
            Encode Captions
        """)
    parser.add_argument("-m", "--model_dir",
                        help="directory of model")
    parser.add_argument("-img-dir", "--image_root_dir", help="root directory of images")
    parser.add_argument("-s2", "--stage_2_dir",
                        help="directory output of stage 02, accessible via network by every node", required=True)
    parser.add_argument(
        "-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
    
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    parquet_dir = Path(f"{args.stage_2_dir}/captions")
    output_path = Path(args.output_dir)
    image_root_dir = Path(args.image_root_dir)

    # Load manifest (caption file with image pair path)
    records = pd.read_parquet(
        parquet_dir,
        engine="pyarrow",
        columns=[
            "document_id",
            "caption",
            "image_path",
            "image_type",
            "first_level_dir",
            "second_level_dir",
            "fit_context",
            "image_file_exist"
        ])

    # Prepare output directory
    if(output_path.exists()):
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    model = AutoModel.from_pretrained(model_dir)
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    DATA_SIZE = records.shape[0]
    BATCH_SIZE = 2048

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Get original index
    records["original_index"] = range(records.shape[0])

    # Select data based on data size
    records = records[records["original_index"] < DATA_SIZE]

    letters = "0123456789abcdef"
    first_level_dir_list = ["{}{}".format(i, j) for i in letters for j in letters]

    for first_level_dir in first_level_dir_list:
        start_time = time.time()
        encode_section(
            records[records["first_level_dir"] == first_level_dir],
            first_level_dir,
            model,
            processor,
            BATCH_SIZE,
            device,
            image_root_dir,
            output_path
        )
        print("{} : ---{:.2f}--- seconds".format(first_level_dir, time.time() - start_time))
    

if __name__ == "__main__":
    main()
