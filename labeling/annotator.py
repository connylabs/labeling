import abc
import random
from typing import Union
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset, Image, concatenate_datasets

from labeling.samplers import Sampler


def make_thumbnail(sample, dim=500, name="thumbnail"):
    sample[name] = sample["image"].copy()
    sample[name].thumbnail((dim, dim))
    return sample


def make_tiny(sample, dim=70, name="tiny"):
    return make_thumbnail(sample, dim=dim, name=name)


def update_label(sample, sample_idx, label=None, idx=None):
    if sample_idx == idx:
        sample["label"] = label
    return sample


def prepare_dump(sample):
    dump = {}
    dump["file_name"] = sample["image"]["path"]
    dump["label"] = sample["label"]
    return dump


def clean_path(sample, dir_name=None):
    dir_name = Path(dir_name).parts[-1]
    sample["file_name"] = Path(sample["file_name"].split(dir_name)[-1]).name
    return sample


def get_index(sample, idx):
    sample["index"] = idx
    return sample


def is_unlabeled(sample):
    return sample["label"] is None

def is_labeled(sample):
    return sample["label"] is not None


class Annotator:
    def __init__(
            self,
            dataset: Union[Dataset, list[dict]],
            sampler: Sampler,
            output_path: Union[str, Path]
        ):
        self.sampler = sampler
        self.output_path = Path(output_path)

        dataset = Dataset.from_list(dataset)
        dataset = dataset.map(make_thumbnail)
        dataset = dataset.map(make_tiny)
        dataset = dataset.cast_column("thumbnail", Image())
        dataset = dataset.cast_column("tiny", Image())

        self.labeled_data = dataset.filter(is_labeled)
        self.unlabeled_data = dataset.filter(is_unlabeled)

        # use sampler to sort
        self.unlabeled_data, scores = self.sampler(self.unlabeled_data)
        self.unlabeled_data = self.unlabeled_data.add_column("score", scores)

        # convert to list of dict
        self.labeled_data = self.labeled_data.to_pandas().to_dict(orient="records")
        self.unlabeled_data = self.unlabeled_data.to_pandas().to_dict(orient="records")

    def __len__(self):
        return len(self.labeled_data) + len(self.unlabeled_data)

    def __getitem__(self, index):
        return (self.labeled_data + self.unlabeled_data)[index]

    @property
    def current_sample(self):
        return self.unlabeled_data[0]

    def redo(self, index):
        current_sample = self.labeled_data.pop(index)
        self.unlabeled_data.insert(0, current_sample)
        return self

    @property
    def dataset(self):
        return concatenate_datasets([
            self.labeled_dataset,
            self.unlabeled_dataset
        ])

    @property
    def labeled_dataset(self):
        return Dataset.from_list(self.labeled_data)

    @property
    def unlabeled_dataset(self):
        return Dataset.from_list(self.unlabeled_data)

    def update(self, label):
        current_sample = self.unlabeled_data.pop(0)
        current_sample["label"] = label
        self.labeled_data.append(current_sample)

        self.to_jsonl()
        self.sampler = self.sampler.update(self.labeled_data)
        return self

    def to_jsonl(self):
        dataset = self.dataset
        drop_cols = [col for col in dataset.column_names if col not in ["file_name", "label"]]
        dump = dataset.cast_column('image', Image(decode=False))
        dump = dump.map(prepare_dump, remove_columns=drop_cols)

        dir_name = self.output_path.parent
        dump = dump.map(clean_path, fn_kwargs={"dir_name": dir_name})
        dump.to_json(self.output_path)#(Path(dir_name).joinpath("metadata.jsonl"))
        return self
