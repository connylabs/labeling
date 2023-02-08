import typing
import random
from pathlib import Path
import contextlib

import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import open as open_image

from datasets import (
    Dataset,
    load_dataset,
    Image,
    concatenate_datasets,
    Features,
    ClassLabel,
    Value
)

from labeling import defaults


def set_random_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)


def to_dataset(
    dataset: list[typing.Dict[str, list[str]]],
    labels: list[str]
) -> Dataset:
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(num_classes=len(labels), names=labels),
            "score": Value("float"),
        }
    )
    return Dataset.from_list(dataset, features=features)


def _dataset_to_list(dataset: Dataset) -> list[typing.Dict[str, list[str]]]:
    dataset = dataset.to_pandas().replace(to_replace=np.nan, value=None)
    return dataset.to_dict(orient="records")


def load_dataset_from_disk(
    dir_name: str,
    metadata_path: str,
    labels: list[str]
) -> list[typing.Dict[str, list[str]]]:

    # define the features in our dataset
    cl = ClassLabel(num_classes=len(labels), names=labels)
    features = Features(
        {
            "image": Image(decode=False),
            "label": cl,
            "score": Value("float"),
        }
    )

    # get the image samples
    dataset = load_dataset(
        "imagefolder",
        data_dir=dir_name,
        split="train",
        features=features
    )

    dataset = _dataset_to_list(dataset)

    # get the metadata
    metadata_path = Path(metadata_path)
    if metadata_path.exists():

        if ".csv" in metadata_path.suffix.lower():
            metadata = pd.read_csv(metadata_path)
        elif ".json" in metadata_path.suffix.lower():
            metadata = pd.read_json(metadata_path, orient="records", lines=True)
        else:
            raise ValueError(f"Expected metadata to be either `jsonl` or `csv` but found {metadata_path}")

        if "label" not in metadata.columns or "file_name" not in metadata.columns:
            raise ValueError(f"Expected metadata to contain `file_name` and `label` columns but found {metadata.columns}")

        metadata = metadata.dropna().replace(to_replace=np.nan, value=None)
        metadata = metadata.set_index("file_name")["label"]

        # metadata = metadata.apply(cl.str2int)
        metadata = metadata.to_dict()

        # add the metadata for labeled samples
        for sample in dataset:
            sample["label"] = metadata.get(sample["image"]["path"], None)

            if isinstance(sample["label"], (float, int)):
                sample["label"] = cl.int2str(sample["label"])

    return dataset


def load_image(image_path, size=500, name="thumbnail"):
    image = open_image(image_path)

    if size is not None:
        image.thumbnail((size, size))

    return image

def make_tiny(image_path, size=70, name="tiny"):
    return load_image(image_path, size=size, name=name)


def prepare_dump(sample, dir_name=None, relative=False):
    sample_ = {}
    sample_["label"] = sample["label"]

    # get the path from the image dict
    sample_["file_name"] = sample["image"]["path"]

    # keep only the filename relative to the directory
    if relative and (dir_name is not None):
        dir_name = Path(dir_name).parts[-1]
        sample_["file_name"] = Path(sample_["file_name"].split(dir_name)[-1]).name

    return sample_

def is_unlabeled(sample):
    return sample["label"] is None

def is_labeled(sample):
    return sample["label"] is not None

def is_not_skipped(sample):
    return sample["label"] is not defaults.SKIP_LABEL


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    copied from https://stackoverflow.com/a/58936697/5257074
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
