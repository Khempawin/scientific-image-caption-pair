import pandas as pd
import re
from typing import TypedDict

input_file_path_template = "/home/horton/datasets/meta-scir/dataset-plain/{}.parquet"
output_file_path_template = "/home/horton/datasets/meta-scir/dataset-meta-special-token/{}.parquet"

lookup_mesh_data = "/home/horton/datasets/meta-scir/lookup_parquet"
# Known sections, ["introduction", "method", "result", "discussion", "conclusion"]

def classify_section(val):
    if("intro" in val):
        return "introduction"
    elif("method" in val):
        return "methods"
    elif("result" in val or "discussion" in val):
        return "results and discussion"
    elif("conclusion" in val):
        return "conclusion"
    else:
        return "other"

# load lookup data
lookup_df = pd.read_parquet(lookup_mesh_data, engine="pyarrow")

lookup_dict = dict()

for i, v in lookup_df.iterrows():
    concepts = [concept["display_name"] for concept in v["concepts"]]
    lookup_dict[v["document_id"]] = concepts

splits = ["train", "val", "test"]

for spilt in splits:
    df = pd.read_parquet(input_file_path_template.format(spilt), engine="pyarrow")
    print("{} : {} records".format(spilt, df.shape[0]))
    print(df["first_level_dir"].unique())
    # Preprocess section 
    df["section"] = df.apply(lambda row: re.sub(r"\s+", " ", row["section"].strip()).lower() if isinstance(row["section"], str) else "", axis=1)

    df["new_section"] = df.apply(lambda row: classify_section(row["section"]), axis=1)

    df["concepts"] = df.apply(lambda row: " , ".join(lookup_dict.get(row["document_id"])) if row["document_id"] in lookup_dict else "", axis=1)

    section_new_count = df.groupby(by="new_section").count()
    print(section_new_count[["caption"]])

    df["original_caption"] = df["caption"]
    df["caption"] = df.apply(lambda row: f"[START-TITLE] {row['article_title']} [END-TITLE] [START-CONCEPT] {row['concepts']} [END-CONCEPT] [START-SECTION] {row['new_section']} [END-SECTION] {row['caption']}", axis=1)
    print(df.head())
    print(df[df["concepts"] == ""].shape)
    
    df.to_parquet(output_file_path_template.format(spilt), engine="pyarrow")