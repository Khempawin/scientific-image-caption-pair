import torch
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple

VALID_FIELDS = {"caption", "document_id", "image_path", "image_type", "fit_context", "first_level_dir", "second_level_dir"}


class ScientificImageCaptionDataset(Dataset):
    def get_full_path(
            self,
            image_path: str, 
            first_level_dir: str, 
            second_level_dir: str, 
            image_root: str) -> str:
        return f"{image_root}/output_{first_level_dir}/{first_level_dir}_{second_level_dir}_images/{image_path}"
    
    def __init__(self, 
                 parquet_file_path: str, 
                 image_root_dir: str, 
                 transform=None):
        self.manifest = pd.read_parquet(parquet_file_path, engine="pyarrow")
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.default_img_transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor()
        ])
        self.expand_img_channel = transforms.Compose([
            transforms.Grayscale(num_output_channels=3)            
        ])

    def __len__(self) -> int:
        return self.manifest.shape[0]
    
    def load_image(self, index: int) -> Image.Image:
        """Opens an image via a path and returns it."""
        record = self.manifest.iloc[index]
        image_path = self.get_full_path(
            image_path=record["image_path"], 
            first_level_dir=record["first_level_dir"], 
            second_level_dir=record["second_level_dir"],
            image_root=self.image_root_dir
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
                first_level_dir=record["first_level_dir"], 
                second_level_dir=record["second_level_dir"],
                image_root=self.image_root_dir
            )
            image = self.load_image(index)
        except:
            raise Exception(f"Error read image path : {image_path}")
        caption = self.get_field(index, "caption")
        document_id = self.get_field(index, "document_id")

        image = self.transform(image) if self.transform else self.default_img_transform(image)

        return (image, caption, document_id)
    