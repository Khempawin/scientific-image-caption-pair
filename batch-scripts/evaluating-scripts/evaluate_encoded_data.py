import faiss
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from typing import TypedDict, Optional, List, Any, Literal
from sklearn.metrics import accuracy_score

VECTOR_DIMENSION = 512

class DatasetConfig(TypedDict):
    name: str
    encoded_data_path: str

class I2TResult(TypedDict):
    base_model: str
    variation: str
    accuracy: float
    records: pd.DataFrame

data_list = [
    # DatasetConfig(name="miread-zero-shot", encoded_data_path="miread-zero-shot.parquet"),
    # DatasetConfig(name="miread-finetuned", encoded_data_path="miread-finetuned.parquet"),
    # DatasetConfig(name="miread-section", encoded_data_path="miread-section.parquet"),
    # DatasetConfig(name="miread-meta", encoded_data_path="miread-meta.parquet"),
    # DatasetConfig(name="miread-special-token-base", encoded_data_path="miread-special-token-base.parquet"),
    # DatasetConfig(name="miread-special-token-meta", encoded_data_path="miread-special-token-meta.parquet"),

    # DatasetConfig(name="roberta-zero-shot", encoded_data_path="roberta-zero-shot.parquet"),
    # DatasetConfig(name="roberta-finetuned", encoded_data_path="roberta-finetuned.parquet"),
    # DatasetConfig(name="roberta-section", encoded_data_path="roberta-section.parquet"),
    # DatasetConfig(name="roberta-meta", encoded_data_path="roberta-meta.parquet"),
    # DatasetConfig(name="roberta-special-token-base", encoded_data_path="roberta-special-token-base.parquet"),
    # DatasetConfig(name="roberta-special-token-meta", encoded_data_path="roberta-special-token-meta.parquet"),

    # DatasetConfig(name="scibert-zero-shot", encoded_data_path="scibert-zero-shot.parquet"),
    # DatasetConfig(name="scibert-finetuned", encoded_data_path="scibert-finetuned.parquet"),
    # DatasetConfig(name="scibert-section", encoded_data_path="scibert-section.parquet"),
    # DatasetConfig(name="scibert-meta", encoded_data_path="scibert-meta.parquet"),
    # DatasetConfig(name="scibert-special-token-base", encoded_data_path="scibert-special-token-base.parquet"),
    # DatasetConfig(name="scibert-special-token-meta", encoded_data_path="scibert-special-token-meta.parquet"),

    # DatasetConfig(name="specter2-zero-shot", encoded_data_path="specter2-zero-shot.parquet"),
    # DatasetConfig(name="specter2-finetuned", encoded_data_path="specter2-finetuned.parquet"),
    # DatasetConfig(name="specter2-section", encoded_data_path="specter2-section.parquet"),
    # DatasetConfig(name="specter2-meta", encoded_data_path="specter2-meta.parquet"),
    # DatasetConfig(name="specter2-special-token-base", encoded_data_path="specter2-special-token-base.parquet"),
    # DatasetConfig(name="specter2-special-token-meta", encoded_data_path="specter2-special-token-meta.parquet"),
    # DatasetConfig(name="specter2-special-token-meta-full", encoded_data_path="specter2-special-token-meta-full.parquet"),
    # DatasetConfig(name="specter2-special-token-meta-full-b192", encoded_data_path="specter2-special-token-meta-full-b192.parquet"),
    # DatasetConfig(name="specter2-meta-b192", encoded_data_path="specter2-meta-b192.parquet"),
    DatasetConfig(name="specter2-meta-b192-full", encoded_data_path="specter2-meta-full-b192.parquet"),
    # DatasetConfig(name="specter2-special-token-meta-b192", encoded_data_path="specter2-special-token-meta-b192.parquet"),
]

def retrieve_text(index, encoded_image_list, k=4):
    search_key = np.asarray(encoded_image_list)
    faiss.normalize_L2(search_key)
    # Search
    _, ann = index.search(search_key, k=k)
    return ann

def build_classifier_index(encoded_data_list: List[Any], device_type: Literal["cpu", "gpu"]):
    vectors = np.asarray(encoded_data_list)
    faiss.normalize_L2(vectors)

    # Create index
    if(device_type == "cpu"):
        index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    elif(device_type == "gpu"):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, VECTOR_DIMENSION, flat_config)

    # Build classifier
    index.add(vectors)
    
    return index

def parse_concepts_2_set(concept_string: str) -> set:
    return set([token.strip() for token in concept_string.split(",")])

def calculate_set_relevance(a: set, b: set) -> float:
    a_union_b = a.union(b)
    union_len = len(a_union_b)
    if(union_len <= 0):
        logging.error("Division by 0")
        pass
    return len(a.intersection(b))/union_len if union_len > 0 and len(a) > 0 else 1

def evaluate_data(base_input_path: str, data_config: DatasetConfig, batch_size: int, data_size_limit: int=0, relevance_metric: Literal["identity", "section", "concepts"] = "identity") -> I2TResult:
    '''
    Expected columns : 'index', 'raw_index', 'document_id', 'caption', 'original_image_path',
       'image_type', 'first_level_dir', 'second_level_dir', 'section',
       'journal_name', 'article_title', 'subjects', 'authors', 'image_path',
       'new_section', 'concepts', 'original_caption', 'encoded_caption',
       'encoded_image', 'load_status'
    '''

    # Load encoded data as dataframe
    records = pd.read_parquet("{}/{}".format(base_input_path, data_config["encoded_data_path"]), engine="pyarrow")
    records = records.reset_index().rename(columns={"index":"raw_index"}).reset_index()

    
    if(data_size_limit != 0):
        records = records.iloc[:data_size_limit]

    # Filter records without concepts
    # records = records[records["concepts"] != ""]

    DATA_SIZE = records.shape[0]
    
    logging.debug("Records with blank concepts: {} / {}".format(records[records["concepts"] == ""].shape[0], DATA_SIZE))

    # Divide records into batches
    batches = [list(range(i, i+batch_size)) if i+batch_size < DATA_SIZE else list(range(i, DATA_SIZE)) for i in range(0, DATA_SIZE, batch_size)]

    # Create indexes
    index = build_classifier_index(list(records["encoded_caption"]), device_type="gpu")

    retrieve_text_indices = list()

    for _, batch in tqdm(enumerate(batches), total=len(batches)): #enumerate(batches):#
        encoded_image_list = list(records.iloc[batch[0]:batch[-1]+1]["encoded_image"])
        retrieved_text = retrieve_text(index=index, encoded_image_list=encoded_image_list, k=4)
        retrieve_text_indices.extend(retrieved_text)

    records["i2t-result"] = retrieve_text_indices
    records["match_top_1"] = records.apply(lambda row: row["index"] == row["i2t-result"][0], axis=1)
    records["i2t-top-1"] = records.apply(lambda row: row["i2t-result"][0], axis=1)
    records["i2t-top-1-section"] = records.apply(lambda row: records.iloc[row["i2t-top-1"]]["new_section"], axis=1)
    records["i2t-top-1-concepts"] = records.apply(lambda row: records.iloc[row["i2t-top-1"]]["concepts"], axis=1)

    # print(records[["index", "raw_index", "i2t-result", "match_top_1", "i2t-top-1"]].head(8))
    # print("{} : {:.4f}".format(data_config["name"], accuracy_score(records["index"], records["i2t-top-1"])))
    
    name_tokens = data_config["name"].split("-")

    accuracy = 0
    if(relevance_metric == "identity"):
        accuracy = accuracy_score(records["index"], records["i2t-top-1"])
    elif(relevance_metric == "section"):
        accuracy = accuracy_score(records["new_section"], records["i2t-top-1-section"])
    elif(relevance_metric == "concepts"):
        accuracy = sum(records.apply(lambda row: calculate_set_relevance(parse_concepts_2_set(row["concepts"]), parse_concepts_2_set(row["i2t-top-1-concepts"])), axis=1)) / DATA_SIZE
    else:
        raise NotImplementedError("Unknown Relevance metric")

    result = I2TResult(
        base_model=name_tokens[0],
        variation="-".join(name_tokens[1:]),
        accuracy=accuracy,
        records=records
    )

    return result

def main():
    base_input_path = "/home/horton/datasets/meta-scir/encoded_data/test"
    # output_path = "/home/horton/datasets/meta-scir/information-retrieval"

    # Initialize log file
    logging.basicConfig(filename="result.out",
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(processName)s %(message)s")
    logging.info("Begin Evaluation")

    print("Test Accuracy : {}".format(accuracy_score([1], [1])))
    metric_list = ("identity", "section", "concepts", ) # "concepts" 
    for metric in metric_list:
        logging.info("Metric : {}".format(metric))
        result_list = list()
        for data_config in data_list:
            result_dict = evaluate_data(
                base_input_path=base_input_path, 
                data_config=data_config, 
                batch_size=512,
                data_size_limit=0,
                relevance_metric=metric
            )
            result_list.append(result_dict)
        result_df = pd.DataFrame(result_list)
        logging.info(result_df.pivot_table("accuracy", index="base_model", columns="variation"))
    

if(__name__ == "__main__"):
    main()
