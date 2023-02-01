import numpy as np
import scipy

from datasets import load_metric
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from PIL.Image import Image
import click

"""
Borrowing heavily from HF turotial
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb
"""


def get_model_transforms(feature_extractor):
    normalize = Normalize(
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std
    )

    train_transforms = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )
    return {"train": train_transforms, "validation": val_transforms}


def make_preprocessor(transforms):
    def preprocessor(example_batch):
        example_batch["pixel_values"] = [
            transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch
    return preprocessor


def get_label_maps(dataset=None, labels=None):
    assert (dataset is not None) or (labels is not None)

    if dataset is not None:
        labels = dataset.features["label"].names

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def collate_train_eval(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def collate_pred(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


class Model:
    def __init__(
        self,
        labels,
        project_name="custom-labeling-project",
        model_checkpoint="microsoft/resnet-18",
        metric_name="f1",
        num_epochs=3,
        batch_size=10,
        test_size=0.2,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        random_seed=42,
        ):

        self.labels = labels
        self.project_name = project_name
        self.model_checkpoint = model_checkpoint
        self.metric_name = metric_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.random_seed = random_seed

        # begin transformations of init args
        self.model = None
        self.metric = load_metric(metric_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        self.transforms = get_model_transforms(self.feature_extractor)

        # prepare dataset
        self.label2id, self.id2label = get_label_maps(labels=labels)

        model_name = model_checkpoint.split("/")[-1]

        self.training_args = TrainingArguments(
            f"{model_name}-finetuned-{project_name}",
            remove_unused_columns=False,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            warmup_ratio=0.1,
            logging_steps=1,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=False,
            seed=random_seed,
        )

    def compute_metrics(self, eval_pred):
        """Computes metrics on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def compute_f1_metric(self, eval_pred):
        """Computes F1 on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")

    def _init_model(self):
        model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )
        return model

    def fit(self, dataset):
        # always fit from scratch
        self.model = self._init_model()

        splits = dataset.train_test_split(test_size=self.test_size)
        train_ds = splits['train']
        val_ds = splits['test']

        train_ds.set_transform(make_preprocessor(self.transforms["train"]))
        val_ds.set_transform(make_preprocessor(self.transforms["validation"]))

        self.trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.feature_extractor,
            compute_metrics=self.compute_f1_metric if self.metric_name.lower() == "f1" else self.compute_metrics,
            data_collator=collate_train_eval,
        )

        train_results = self.trainer.train()

        return self

    def predict_proba(self, dataset):
        # allow predicting from un-trained model
        if self.model is None:
            self.model = self._init_model()

        trainer = Trainer(
            self.model,
            self.training_args,
            tokenizer=self.feature_extractor,
            data_collator=collate_pred,
        )

        # data validation
        dataset.set_transform(make_preprocessor(self.transforms["validation"]))

        preds = trainer.predict(dataset)
        logits = preds.predictions
        return scipy.special.softmax(logits, axis=-1)

    def predict(self, dataset):
        probs = self.predict_proba(dataset).tolist()
        # sort by labels by predicted probs
        out = []
        for probs_ in probs:
            out_ = [
                {"label":key, "score":prob} for key, prob in zip(self.label2id, probs_)
            ]

            out_ = sorted(
                out_,
                key=lambda pred: pred["score"],
                reverse=True
            )

            out.append(out_)
        return out
