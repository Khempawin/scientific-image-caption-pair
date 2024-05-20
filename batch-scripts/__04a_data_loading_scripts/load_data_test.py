from datasets import load_dataset

dataset_name = ""

dataset = load_dataset(
    "./load_data_scir.py", 
    data_dir="/home/horton/datasets/meta-scir/dataset-plain", 
    trust_remote_code=True,
    split=["train", "validation", "test"]
    )

print("COMPLETED : PLAIN")

dataset = load_dataset(
    "./load_data_scir.py", 
    data_dir="/home/horton/datasets/meta-scir/dataset-section", 
    trust_remote_code=True,
    split=["train", "validation", "test"]
    )
print("COMPLETED : SECTION")

dataset = load_dataset(
    "./load_data_scir.py", 
    data_dir="/home/horton/datasets/meta-scir/dataset-title-section", 
    trust_remote_code=True,
    split=["train", "validation", "test"]
    )
print("COMPLETED : TITLE-SECTION")

dataset = load_dataset(
    "./load_data_scir.py", 
    data_dir="/home/horton/datasets/meta-scir/dataset-meta-special-token", 
    trust_remote_code=True,
    split=["train", "validation", "test"]
    )
print(len(dataset[0]))
print("COMPLETED : META-SPECIAL-TOKEN")