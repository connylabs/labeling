import abc
from pathlib import Path
import random

import numpy as np
from datasets import Dataset, DatasetDict

from labeling.hf_model import Model, get_label_maps


def set_random_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dataset: Dataset) -> [Dataset, np.array]:
        pass

    @abc.abstractmethod
    def update(self, dataset: Dataset):
        return self


class Sampler(BaseSampler):
    @abc.abstractmethod
    def update(self, dataset: Dataset) -> BaseSampler:
        return self


class RandomSampler(Sampler):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        set_random_seed(random_seed)

    def __call__(self, dataset):
        return dataset.shuffle(seed=self.random_seed), np.zeros(len(dataset))

    def update(self, dataset: Dataset) -> BaseSampler:
        return self


class ActiveLearningSampler(Sampler):
    def __init__(
        self,
        dataset=None,
        labels=None,
        batch_size=10
        ):
        assert (dataset is not None) or (labels is not None)

        self.batch_size = batch_size
        self.labels = labels or dataset.features["label"].names
        self.dataset = dataset

        self.model = Model(self.labels, batch_size=batch_size)
        self.updates = 0

        if dataset is not None:
            self.fit(dataset)

    def __call__(self, dataset):
        preds = self.predict(dataset)
        scores = [pred["score"] for pred in preds]
        scores = np.array(scores)
        if scores.ndim == 1:
            scores = scores[:, None]
            scores = np.hstack([scores, 1-scores])
        if scores.ndim > 2:
            raise ValueError(f"Expected prediction scores to have dim 2, but found {scores}")

        entropies = np.mean(-np.log(scores) * scores, axis=-1)
        idxs = np.argsort(entropies, axis=-1)[::-1] # get idxs of samples with max entrpoy(
        return dataset.select(idxs), entropies[idxs]

    def fit(self, dataset):
        if not isinstance(dataset, (Dataset, DatasetDict)):
            pass
        self.model.fit(dataset)
        return self

    def predict(self, dataset):
        return self.model.predict(dataset)

    def update(self, dataset):
        self.updates += 1
        if (self.updates % self.batch_size) == 0:
            return self.fit(dataset)
        return self


SAMPLERS = {
    "active-learning": ActiveLearningSampler,
    "random": RandomSampler
}
