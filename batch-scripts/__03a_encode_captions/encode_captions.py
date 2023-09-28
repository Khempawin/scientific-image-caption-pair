import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from transformers import (
    VisionTextDualEncoderProcessor,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
)
from PIL import Image

# Used to enable VisionTextDualModel encoding of Text
DUMMY_IMAGE = Image.fromarray(np.random.randint(255, size=(224,224,3),dtype=np.uint8))




def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="""
            Encode captions
        """)
    parser.add_argument("-s2", "--stage_2_dir",
                        help="directory output of stage 02, accessible via network by every node", required=True)
    parser.add_argument(
        "-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
    
    args = parser.parse_args()
    parquet_dir = f"{args.stage_2_dir}/captions"
    output_dir = args.output_dir

    # Load data manifest
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(records.shape)
    print(records.head())
    # selected.compute().to_parquet(output_path / "captions",
    #                               engine="pyarrow", partition_cols="first_level_dir")


if __name__ == "__main__":
    main()
