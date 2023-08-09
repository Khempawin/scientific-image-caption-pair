import torch
import pandas as pd
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
                 transform=None):
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.manifest = self.load_manifest(parquet_file_path)

    def __len__(self) -> int:
        return self.manifest.shape[0]
    
    def check_img_exists(self, record) -> bool:
        image_path = Path(self.get_full_path(record["image_path"], record["first_level_dir"]))
        return image_path.is_file()
    
    def load_manifest(self, parquet_file_path: str):
        manifest = pd.read_parquet(parquet_file_path, engine="pyarrow")
        image_ready = manifest.apply(lambda record: self.check_img_exists(record), axis=1) 
        return manifest[image_ready]
    
    def load_image(self, index: int) -> Image.Image:
        """Opens an image via a path and returns it."""
        record = self.manifest.iloc[index]
        image_path = self.get_full_path(
            image_path=record["image_path"], 
            first_level_dir=record["first_level_dir"]
        )
        return Image.open(image_path).convert("RGB")
    
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
            image = self.load_image(index)
        except:
            raise Exception(f"Error read image path : {image_path}")
        caption = self.get_field(index, "caption")
        document_id = self.get_field(index, "document_id")

        image = self.transform(image) if self.transform else image

        return (image, caption, document_id)
    