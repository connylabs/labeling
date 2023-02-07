from __future__ import annotations
import abc
from pathlib import Path
from copy import copy

import numpy as np
import scipy
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image

from labeling.model import Model, get_label_maps, MODEL
from labeling.utils import SKIP_LABEL, to_dataset, _dataset_to_list


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dataset: Dataset) -> [Dataset, np.array]:
        pass

    @abc.abstractmethod
    def step(self, dataset: Dataset) -> tuple(BaseSampler, bool):
        return self, False


class RandomSampler(BaseSampler):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        set_random_seed(random_seed)

    def __call__(self, dataset):
        return dataset.shuffle(seed=self.random_seed), np.zeros(len(dataset))

    def step(self, dataset: Dataset) -> tuple(BaseSampler, bool):
        return self, False


class ActiveLearningSampler(BaseSampler):
    def __init__(self, labels, retrain_steps=10, model_name=MODEL):
        self.labels = copy(labels)
        self.labels.remove(SKIP_LABEL)
        self.retrain_steps = retrain_steps
        self.updates = 0
        self.model_name = model_name or MODEL

    def __call__(self, dataset):
        return self.sort(dataset)

    @property
    def is_fitted(self):
        return hasattr(self, "model")

    def sort(self, dataset):
        if not self.is_fitted:
            raise ValueError("Sampler is not yet fitted. Run `sampler.fit(...)` with your dataset first.")

        if not isinstance(dataset, (Dataset, DatasetDict)):
            dataset = to_dataset(dataset, labels=self.labels)

        scores = self.model.predict_proba(dataset)
        if scores.ndim == 1:
            scores = scores[:, None]
            scores = np.hstack([scores, 1-scores])

        if scores.ndim > 2:
            raise ValueError(f"Expected prediction scores to have dim 2, but found {scores}")

        entropies = scipy.stats.entropy(scores, axis=-1)
        idxs = np.argsort(entropies, axis=-1) # highest entropies last because we will pop from list end
        return _dataset_to_list(dataset.select(idxs)), entropies[idxs]

    def fit(self, dataset):
        if not isinstance(dataset, (Dataset, DatasetDict)):
            dataset = to_dataset(dataset, labels=self.labels)

        self.model = Model(self.labels, model_checkpoint=self.model_name)

        self.model.fit(dataset)
        return self

    def step(self, dataset):
        self.updates += 1
        if (self.updates % self.retrain_steps) == 0:
            self.fit(dataset)
            return self, True
        return self, False


SAMPLERS = {
    "active-learning": ActiveLearningSampler,
    "random": RandomSampler
}
