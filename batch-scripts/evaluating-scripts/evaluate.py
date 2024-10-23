from typing import TypedDict
from transformers import (
    VisionTextDualEncoderProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
)
import pandas as pd
from zipfile import ZipFile
from PIL import Image
import logging
import io
import torch
import numpy as np
from tqdm import tqdm

class ModelConfig(TypedDict):
    model_name: str
    model_dir: str

model_list = [
    # ModelConfig(model_name="zero-shot-miread", model_dir="/home/horton/datasets/meta-scir/models/miread"),
    # ModelConfig(model_name="zero-shot-roberta", model_dir="/home/horton/datasets/meta-scir/models/roberta"),
    # ModelConfig(model_name="zero-shot-scibert", model_dir="/home/horton/datasets/meta-scir/models/scibert"),
    # ModelConfig(model_name="zero-shot-specter2", model_dir="/home/horton/datasets/meta-scir/models/specter2"),
    # ModelConfig(model_name="clip-miread", model_dir="/home/horton/datasets/meta-scir/models/miread-test"),
    # ModelConfig(model_name="clip-roberta", model_dir="/home/horton/datasets/meta-scir/models/roberta-test"),
    # ModelConfig(model_name="clip-scibert", model_dir="/home/horton/datasets/meta-scir/models/scibert-test"),
    # ModelConfig(model_name="clip-specter2", model_dir="/home/horton/datasets/meta-scir/models/specter2-test"),
    # ModelConfig(model_name="section-miread", model_dir="/home/horton/datasets/meta-scir/models/miread-section"),
    # ModelConfig(model_name="section-roberta", model_dir="/home/horton/datasets/meta-scir/models/roberta-section"),
    # ModelConfig(model_name="section-scibert", model_dir="/home/horton/datasets/meta-scir/models/scibert-section"),
    # ModelConfig(model_name="section-specter2", model_dir="/home/horton/datasets/meta-scir/models/specter2-section"),
    # ModelConfig(model_name="meta-miread", model_dir="/home/horton/datasets/meta-scir/models/miread-meta"),
    # ModelConfig(model_name="meta-roberta", model_dir="/home/horton/datasets/meta-scir/models/roberta-meta"),
    # ModelConfig(model_name="meta-scibert", model_dir="/home/horton/datasets/meta-scir/models/scibert-meta"),
    # ModelConfig(model_name="meta-specter2", model_dir="/home/horton/datasets/meta-scir/models/specter2-meta"),
    ModelConfig(model_name="meta-specter2-special-token-base", model_dir="/home/horton/datasets/meta-scir/models/specter2-special-token-base"),
]

test_data_manifest_path = "/home/horton/datasets/meta-scir/dataset-title-section/test.parquet"
test_data_image_zip_path = "/home/horton/datasets/meta-scir/dataset-title-section/test.zip"

def load_model(model_config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_dir"])
    image_processor = CLIPImageProcessor.from_pretrained(model_config["model_dir"])

    model = AutoModel.from_pretrained(model_config["model_dir"])
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    return model, processor

def load_image_from_zip_file(image_zip:ZipFile, image_path:str) -> Image.Image:
    img_file = image_zip.read(image_path)
    img = Image.open(io.BytesIO(img_file)).convert("RGB")
    return True, img

manifest = pd.read_parquet(test_data_manifest_path, engine="pyarrow")
image_zip = ZipFile(test_data_image_zip_path, "r")

print(manifest.columns)

DATA_SIZE = manifest.shape[0]
batch_size = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# Divide records into batches
batches = [list(range(i, i+batch_size)) if i+batch_size < DATA_SIZE else list(range(i, DATA_SIZE)) for i in range(0, DATA_SIZE, batch_size)]

# Encode captions
for model_config in model_list:
    print("Processing : {}".format(model_config["model_name"]))
    model, processor = load_model(model_config)
    model = model.to(device)
    # Initialize encoded lists
    encoded_caption_list = list()
    encoded_image_list = list()
    load_status_list = list()

    # Process each batch
    for batch in tqdm(batches):
        image_list = list()
        # Load images
        for i in batch:
            success, image = load_image_from_zip_file(image_zip, manifest.iloc[i]["image_path"])
            load_status_list.append(success)
            image_list.append(image)
        # Select captions for batch
        caption_list = list(manifest.iloc[batch[0]:batch[-1]+1]["caption"])
        # Encode image and caption
        with torch.no_grad():
            inputs = processor(text=caption_list, images=image_list, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            inputs["input_ids"] = inputs["input_ids"][:,:512]
            inputs["token_type_ids"] = inputs["token_type_ids"][:,:512]
            inputs["attention_mask"] = inputs["attention_mask"][:,:512]
            # print(inputs["input_ids"][:,:512].shape)
            # print(inputs.keys())
            outputs = model(**inputs)
        
        # # Add encoded batch result to encoded lists
        # encoded_caption_list.extend(list(np.asarray(outputs.text_embeds.to("cpu"))))
        # encoded_image_list.extend(list(np.asarray(outputs.image_embeds.to("cpu"))))
        # break
    break
    