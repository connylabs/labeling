import abc
import random
import typing
from pathlib import Path
from functools import partial

import pandas as pd

from labeling.samplers import BaseSampler
from labeling.utils import (
    load_dataset_from_disk,
    is_labeled,
    is_unlabeled,
    is_not_skipped,
    prepare_dump
)


class Annotator:
    def __init__(
            self,
            dataset: list[typing.Dict[str, list[str]]],
            sampler: BaseSampler,
            output_path: typing.Union[str, Path],
            limit=None,
        ):
        self.sampler = sampler
        self.output_path = Path(output_path)
        self.limit = limit

        self.labeled_data = list(filter(is_labeled, dataset))
        self.unlabeled_data = list(filter(is_unlabeled, dataset))[:limit]

        if len(self.trainable_data) > 0:
            self.sampler.fit(self.trainable_data)
            self.unlabeled_data = self.sort(self.unlabeled_data)

    def __len__(self):
        return len(self.labeled_data) + len(self.unlabeled_data)

    @property
    def current_sample(self):
        return self.unlabeled_data[-1]

    @property
    def trainable_data(self):
        return list(filter(is_not_skipped, self.labeled_data))

    def redo(self, index):
        current_sample = self.labeled_data.pop(index)
        self.unlabeled_data.append(current_sample)
        return self

    def set_label(self, label):
        current_sample = self.unlabeled_data.pop()
        current_sample["label"] = label
        self.labeled_data.append(current_sample)

        self.to_jsonl()
        self.sampler, needs_sort = self.sampler.step(self.trainable_data)
        if needs_sort:
            self.unlabeled_data = self.sort(self.unlabeled_data)
        return self

    def sort(self, data: list[typing.Dict[str, list[str]]]):
        # use sampler to sort
        data, scores = self.sampler(data)

        for sample, score in zip(data, scores):
            sample["score"] = score

        return data

    def to_jsonl(self):
        dir_name = self.output_path.parent
        dataset = map(
            partial(prepare_dump, dir_name=dir_name),
            self.labeled_data
        )
        dataset = pd.DataFrame.from_records(dataset)
        dataset.to_json(self.output_path, lines=True, orient="records")
        return self
