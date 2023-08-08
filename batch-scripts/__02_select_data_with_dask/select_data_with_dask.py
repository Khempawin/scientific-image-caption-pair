import dask.dataframe as dd
import argparse
import shutil
from pathlib import Path


def get_image_path(
        base_path: str,
        first_level: str,
        second_level: str,
        image_path: str
):
    return f"{base_path}/output_{first_level}/{first_level}_{second_level}_images/{image_path}"


def process_image_record(row, base_path: str, output_path: str, include_image: bool=False):
    src_path = Path(get_image_path(
        base_path=base_path,
        first_level=row["first_level_dir"],
        second_level=row["second_level_dir"],
        image_path=row["image_path"]
    ))

    if not src_path.is_file():
        return False

    # Prepare image output directory
    if include_image:
        output_image_path = Path(f"{output_path}/image_{row['first_level_dir']}")
        output_image_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=src_path, dst=f"{output_image_path}/{row['image_path']}")
    return True


def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="""
            Select caption records with context length limit 77 and image file exists
        """)
    parser.add_argument("-s0", "--stage_0_dir",
                        help="directory output of stage 00, accessible via network by every node", required=True)
    parser.add_argument("-s1", "--stage_1_dir",
                        help="directory output of stage 01, accessible via network by every node", required=True)
    parser.add_argument(
        "-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
    parser.add_argument(
        "-img", "--image_out", help="copy image from input directory or not", default=False, required=False
    )

    args = parser.parse_args()
    img_dir = args.stage_0_dir
    parquet_dir = args.stage_1_dir
    output_dir = args.output_dir
    image_output = parse_boolean(args.image_out)

    records = dd.read_parquet(
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

    selected = records.query(
        "fit_context == 77 and image_file_exist == True and image_type == 'figure'")

    selected["image_copied"] = selected.apply(lambda row: process_image_record(row, img_dir, output_path, image_output), meta=(None, bool), axis=1)

    selected.compute().to_parquet(output_path / "captions",
                                  engine="pyarrow", partition_cols="first_level_dir")


if __name__ == "__main__":
    main()
