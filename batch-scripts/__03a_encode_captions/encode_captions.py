import dask.dataframe as dd
import argparse
from pathlib import Path


def parse_boolean(value: str):
    if(value and value == "True"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="""
            Select caption records with context length limit 77 and image file exists
        """)
    parser.add_argument("-s2", "--stage_2_dir",
                        help="directory output of stage 02, accessible via network by every node", required=True)
    parser.add_argument(
        "-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
    
    args = parser.parse_args()
    parquet_dir = f"{args.stage_2_dir}/captions"
    output_dir = args.output_dir

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


    # selected.compute().to_parquet(output_path / "captions",
    #                               engine="pyarrow", partition_cols="first_level_dir")


if __name__ == "__main__":
    main()
