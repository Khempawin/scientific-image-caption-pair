import os
import pandas as pd
from pathlib import Path

# Output Directory of Stage01
template_dir = "stage00_image_caption_meta/output_{:02}/caption"

def get_jpg_image_name(old_filename: str):
    return "{}.{}".format(".".join(old_filename.split(".")[:-1]), "jpg")

dir_names = [f"{i}{j}" for i in "0123456789abcdef" for j in "0123456789abcdef"]

output_dir = "./meta-caption-all"
#Make output dir
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

for dir_entry in dir_names:
    df = pd.read_parquet(template_dir.format(dir_entry), engine="pyarrow")\
            .drop_duplicates(["document_id", "caption"])
    df["image_path"] = df.apply(lambda row: get_jpg_image_name(row["image_path"]), axis=1)
    df.to_parquet(output_dir / f"{dir_entry}.parquet.gz", engine="pyarrow")

print(df.columns)