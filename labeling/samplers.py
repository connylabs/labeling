import abc
from pathlib import Path
import random

import numpy as np
from datasets import Dataset

import torch


MODEL = None

def set_random_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    def __init__(self, dataset=None, model=MODEL, batch_size=10):
        super(ActiveLearningSampler, self).__init__(dataset)
        self.model = model
        self.fit(dataset)
        self.batch_size = batch_size
        self.updates = 0

    def __call__(self, dataset):
        scores = self.predict(dataset)
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
        self.model.fit(dataset)
        return self

    def predict(dataset):
        scores = self.model.predict(dataset)
        return scores

    def update(self, dataset):
        self.updates += 1
        if (self.updates % self.batch_size) == 0:
            return self.fit(dataset)
        return self
