"""
PART 1: Data
"""
import numpy as np

from transformations import (
    ComposeDouble,
    FunctionWrapperDouble,
    create_dense_target,
    normalize_01,
)
from customdatasets import SegmentationDataSet1
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib

# root directory
root = pathlib.Path.cwd() / "Carvana"


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / "Input")
targets = get_filenames_of_path(root / "Target")

# training transformations and augmentations
transforms = ComposeDouble(
    [
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs, random_state=random_seed, train_size=train_size, shuffle=True
)

targets_train, targets_valid = train_test_split(
    targets, random_state=random_seed, train_size=train_size, shuffle=True
)

# dataset training
dataset_train = SegmentationDataSet1(
    inputs=inputs_train, targets=targets_train, transform=transforms
)

# dataset validation
dataset_valid = SegmentationDataSet1(
    inputs=inputs_valid, targets=targets_valid, transform=transforms
)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train, batch_size=2, shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2, shuffle=True)


x, y = next(iter(dataloader_training))

print(f"x = shape: {x.shape}; type: {x.dtype}")
print(f"x = min: {x.min()}; max: {x.max()}")
print(f"y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}")


# create DatasetViewer instances
from visual import DatasetViewer

dataset_viewer_training = DatasetViewer(dataset_train)
dataset_viewer_validation = DatasetViewer(dataset_valid)


# open napari instance for training dataset
# navigate with 'n' for next and 'b' for back on the keyboard
dataset_viewer_training.napari()


# open napari instance for validation dataset
# navigate with 'n' for next and 'b' for back on the keyboard
dataset_viewer_validation.napari()

import albumentations
from transformations import AlbuSeg2d


# training transformations and augmentations
transforms_training = ComposeDouble(
    [
        AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)


# validation transformations
transforms_validation = ComposeDouble(
    [
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

