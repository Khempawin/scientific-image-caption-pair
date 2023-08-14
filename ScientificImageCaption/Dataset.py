import torch
import pandas as pd
import dask.dataframe as dd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path

VALID_FIELDS = {"caption", "document_id", "image_path", "image_type", "fit_context", "first_level_dir", "second_level_dir"}


class ScientificImageCaptionDataset(Dataset):
    def get_full_path_s00(
            self,
            image_path: str, 
            first_level_dir: str, 
            second_level_dir: str, 
            image_root: str) -> str:
        return f"{image_root}/output_{first_level_dir}/{first_level_dir}_{second_level_dir}_images/{image_path}"
    
    def get_full_path(
            self,
            image_path: str, 
            first_level_dir: str) -> str:
        return f"{self.image_root_dir}/image_{first_level_dir}/{image_path}"

    def __init__(self, 
                 parquet_file_path: str, 
                 image_root_dir: str, 
                 transform=None,
                 verify_image_file: bool=False):
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.manifest = self.load_manifest(parquet_file_path, verify_image_file)

    def __len__(self) -> int:
        return self.manifest.shape[0]
    
    def has_valid_image_file(self, record) -> bool:
        # check if image exists
        image_path = Path(self.get_full_path(record["image_path"], record["first_level_dir"]))
        if not image_path.is_file():
            return False
        # check if image file can be opened
        if(self.load_image(image_path)):
            return True
        else:
            return False
    
    def load_manifest(self, parquet_file_path: str, verify_image_file: bool=False):
        manifest = pd.read_parquet(parquet_file_path, engine="pyarrow")
        if(verify_image_file):
            image_ready = manifest.apply(lambda record: self.has_valid_image_file(record), axis=1) 
            return manifest[image_ready]
        return manifest
    
    def load_image(self, image_path: str) -> Image.Image | None:
        """Opens an image via a path and returns it."""
        try:
            image = Image.open(image_path).convert("RGB") 
            return image
        except:
            return None 
    
    def get_field(self, index: int, field_name: str) -> str:
        record = self.manifest.iloc[index]

        if field_name not in VALID_FIELDS:
            raise Exception("Invalid field name")

        return record[field_name]
    
    def __getitem__(self, index) -> Tuple[torch.Tensor , str, str]:
        try:
            record = self.manifest.iloc[index]
            image_path = self.get_full_path(
                image_path=record["image_path"], 
                first_level_dir=record["first_level_dir"]
            )
            image = self.load_image(image_path).convert("RGB")
        except:
            raise Exception(f"Error read image path : {image_path}")
        caption = self.get_field(index, "caption")
        document_id = self.get_field(index, "document_id")
        first_level = self.get_field(index, "first_level_dir")
        second_level = self.get_field(index, "second_level_dir")

        image = self.transform(image) if self.transform else image

        return (image, caption, document_id, first_level, second_level)
    