import time
import os
import copy

from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


"""
borrowed heavily from
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#
"""

DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def make_dataloaders(train, val=None, test=None, transforms=DATA_TRANSFORMS, num_workers=4, batch_size=16):
    datasets = {}
    datasets["train"] = Dataset.from_list(train).with_format("torch")

    if val is not None:
        datasets["validation"] = Dataset.from_list(val).with_format("torch")

    if test is not None:
        datasets["test"] = Dataset.from_list(test).with_format("torch")

    for split in datasets:
        if split in transforms:
            datasets[split].set_transform(transforms[split])

    dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=split=="train",
            num_workers=num_workers
        ) for x in datasets.keys()
    }

    return dataloaders


def get_model(n_classes, device=None):
    """Prepare a frozen ResNet18 feature extractor with a trainable linear classifier
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=None):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {split: len(dataloaders[split]) for split in dataloaders}

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
