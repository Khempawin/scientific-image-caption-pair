import dask.dataframe as dd
import argparse

def main():
    parser = argparse.ArgumentParser(description="""
            Select caption records with specified context length limit
        """)
    parser.add_argument("-i", "--input_dir", help="input directory containing parquet files, accessible via network by every node", required=True)
    parser.add_argument("-o", "--output_dir", help="output directory, accessible via network by every node", required=True)
    parser.parse_args()
    input_dir = parser.input_dir
    output_dir = parser.output_dir
    
    records = dd.read_parquet(
        input_dir, 
        engine="pyarrow",
        columns=[
            "document_id",
            "caption",
            "image_path",
            "image_type",
            "first_level_dir",
            "second_level_dir",
            "fit_context"
        ])


    selected = records.query("fit_context == 77")
    selected.compute().to_parquet(output_dir, engine="pyarrow", partition_cols="first_level_dir")

if __name__ == "__main__":
    main()