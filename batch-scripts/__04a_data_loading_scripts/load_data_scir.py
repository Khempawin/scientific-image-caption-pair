import pandas as pd
import os
import datasets


class SCIRBuilderConfig(datasets.BuilderConfig):

    def __init__(self, name, splits, **kwargs):
        super().__init__(name, **kwargs)
        self.splits = splits


# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """WIP"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
SCIR WIP
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# Add the licence for the dataset here if you can find it
_LICENSE = ""

# Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

# This script is supposed to work with local (downloaded) COCO dataset.
_URLs = {}


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class SCIRDataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) SCIR dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = SCIRBuilderConfig
    BUILDER_CONFIGS = [
        SCIRBuilderConfig(name='scir', splits=['train', 'valid', 'test']),
    ]
    DEFAULT_CONFIG_NAME = "scir"

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        feature_dict = {
            # "document_id": datasets.Value("string"),
            # "caption": datasets.Value("string"),
            # "section": datasets.Value("string"),
            # "image_path": datasets.Value("string"),
            # "image_type": datasets.Value("string"),
            # "new_section": datasets.Value("string"),
            # "section_enhanced_caption": datasets.Value("string")


            "image_id": datasets.Value("int64"),
            "caption_id": datasets.Value("int64"),
            "caption": datasets.Value("string"),
            # "height": datasets.Value("int64"),
            # "width": datasets.Value("int64"),
            "file_name": datasets.Value("string"),
            # "coco_url": datasets.Value("string"),
            "image_path": datasets.Value("string"),
            # "new_section": datasets.Value("string"),
            # "section": datasets.Value("string"),

        }

        features = datasets.Features(feature_dict)

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        data_dir = self.config.data_dir
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO dataset. The argument `data_dir` in `load_dataset()` is required."
            )

        _DL_URLS = {
            "train": os.path.join(data_dir, "train.zip"),
            "val": os.path.join(data_dir, "val.zip"),
            "test": os.path.join(data_dir, "test.zip"),
            "manifest_train": os.path.join(data_dir, "train.parquet"),
            "manifest_val": os.path.join(data_dir, "val.parquet"),
            "manifest_test": os.path.join(data_dir, "test.parquet"),
        }
        archive_path = dl_manager.download_and_extract(_DL_URLS)

        splits = []
        for split in self.config.splits:
            if split == 'train':
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "manifest_path": os.path.join(archive_path["manifest_train"]),
                        "image_dir": os.path.join(archive_path["train"]),
                        "split": "train",
                    }
                )
            elif split in ['val', 'valid', 'validation', 'dev']:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "manifest_path": os.path.join(archive_path["manifest_val"]),
                        "image_dir": os.path.join(archive_path["val"]),
                        "split": "valid",
                    },
                )
            elif split == 'test':
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "manifest_path": os.path.join(archive_path["manifest_test"]),
                        "image_dir": os.path.join(archive_path["test"]),
                        "split": "test",
                    },
                )
            else:
                continue

            splits.append(dataset)

        return splits

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self, manifest_path, image_dir, split
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        _features = ["image_id", "caption_id", "caption", "file_name", "image_path", "id"]#, "new_section", "section"] #, "height", "width", "coco_url"
        features = list(_features)

        if split in "valid":
            split = "val"

        data = pd.read_parquet(manifest_path, engine="pyarrow")\
            .reset_index()\
                .drop(columns=["index"])\
                    .reset_index().rename(columns={"index": "id"})

        d = dict()
        # Build dict of image dict
        for _, val in data.iterrows():
            d[val["id"]] = {
                "file_name": val["image_path"],
                "id": val["id"]
            }

        # list of dict
        if split in ["train", "val"]:
            annotations = list()

            # build a dict of image_id ->
            for record in data.to_dict(orient="records"):
                _id = record["id"]
                image_info = d[record["id"]]
                record.update(image_info)
                record["id"] = _id
                record["image_id"] = _id
                annotations.append(record)

            entries = annotations

        elif split == "test":
            entries = list()
            for record in data.to_dict(orient="records"):
                _id = record["id"]
                record["file_name"] = record["image_path"]
                entries.append(record)


        for id_, entry in enumerate(entries):
            entry = {k: v for k, v in entry.items() if k in features}

            if split == "test":
                entry["image_id"] = entry["id"]
                entry["id"] = -1
                entry["caption"] = -1

            entry["caption_id"] = entry.pop("id")
            entry["image_path"] = os.path.join(image_dir, entry["file_name"])

            entry = {k: entry[k] for k in _features if k in entry}

            yield str((entry["image_id"], entry["caption_id"])), entry

            