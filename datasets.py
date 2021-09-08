# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for multi-modal data and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file, padding_image

DATASETS_CONFIG = {
    "Houston2013": {
        "urls": [],
    },
    "Trento":{
        "urls": [],
    },
    "Augsburg":{
        "urls": [],
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get("folder", dataset_name + "/")
    if dataset.get("download", True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for url in datasets[dataset_name]["urls"]:
            # download the files
            filename = url.split("/")[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc="Downloading {}".format(filename),
                ) as t:
                    urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston2013':

        # Load the image
        img1 = open_file(folder + 'HSI.mat')['HSI'].astype(np.float32)
        rgb_bands = (59, 40, 23)

        img2 = open_file(folder + 'LiDAR.mat')['LiDAR'].astype(np.float32)
        img2 = np.expand_dims(img2, axis=2)    # (349, 1905) --> (349, 1905, 1)
        gt = open_file(folder + 'gt.mat')['gt']     # Here, the gt file is load for filtering NaN out and visualization.

        # normalization method 1: map to [0, 1]
        [m, n, l] = img1.shape
        for i in range(l):
            minimal = img1[:, :, i].min()
            maximal = img1[:, :, i].max()
            img1[:, :, i] = (img1[:, :, i] - minimal) / (maximal - minimal)

        minimal = img2.min()
        maximal = img2.max()
        img2 = (img2 - minimal) / (maximal - minimal)

        label_values = [
            "Unclassified",
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track"
        ]

        ignored_labels = [0]

    elif dataset_name == "Trento":
        # Load the image
        img1 = open_file(folder + 'HSI.mat')['HSI'].astype(np.float32)
        rgb_bands = (40, 20, 10)

        img2 = open_file(folder + 'LiDAR.mat')['LiDAR'].astype(np.float32)
        img2 = np.expand_dims(img2, axis=2)  # (600, 166) --> (600, 166, 1)
        gt = open_file(folder + 'gt.mat')['gt']     # Here, the gt file is load for filtering NaN out and visualization.


        # normalization method 1: map to [0, 1]
        [m, n, l] = img1.shape
        for i in range(l):
            minimal = img1[:, :, i].min()
            maximal = img1[:, :, i].max()
            img1[:, :, i] = (img1[:, :, i] - minimal) / (maximal - minimal)

        minimal = img2.min()
        maximal = img2.max()
        img2 = (img2 - minimal) / (maximal - minimal)

        label_values = [
            "Unclassified",
            "Apple trees",
            "Buildings",
            "Ground",
            "Wood",
            "Vineyard",
            "Roads"
        ]

        ignored_labels = [0]

    elif dataset_name == "Augsburg":

        # Load the image
        img1 = open_file(folder + 'data_HS_LR.mat')['data_HS_LR'].astype(np.float32)
        rgb_bands = (40, 20, 10)    # To Do: fix

        img2 = open_file(folder + 'data_DSM.mat')['data_DSM'].astype(np.float32)
        img2 = np.expand_dims(img2, axis=2)  # (332, 485) --> (332, 485, 1)
        gt = open_file(folder + 'gt.mat')['gt']     # Here, the gt file is load for filtering NaN out and visualization.


        # normalization method 1: map to [0, 1]
        [m, n, l] = img1.shape
        for i in range(l):
            minimal = img1[:, :, i].min()
            maximal = img1[:, :, i].max()
            img1[:, :, i] = (img1[:, :, i] - minimal) / (maximal - minimal)

        minimal = img2.min()
        maximal = img2.max()
        img2 = (img2 - minimal) / (maximal - minimal)

        label_values = [
            "Unclassified",
            "Forest",
            "Residential Area",
            "Industrial Area",
            "Low Plants",
            "Allotment",
            "Commercial Area",
            "Water"
        ]

        ignored_labels = [0]

    else:
        # Custom dataset
        (
            img,
            gt,
            rgb_bands,
            ignored_labels,
            label_values,
            palette,
        ) = CUSTOM_DATASETS_CONFIG[dataset_name]["loader"](folder)

    # Filter NaN out
    nan_mask = np.isnan(img1.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img1[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))

    # Normalization
    # img = np.asarray(img, dtype="float32")
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # the shapes of img1 and img2 are both (H, W, C)
    return img1, img2, gt, label_values, ignored_labels, rgb_bands, palette


class MultiModalX(torch.utils.data.Dataset):
    """ Generic class for a MultiModal Data """

    def __init__(self, data, data2, gt, **hyperparams):
        """
        Args:
            data: The first modality (usually 3D hyperspectral image)
            data2: The second modality (usually LiDAR image)
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(MultiModalX, self).__init__()
        self.data = data
        self.data2 =data2
        self.label = gt
        self.name = hyperparams["dataset"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        supervision = hyperparams["supervision"]
        # Fully supervised : use all pixels with label not ignored
        if supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def rotate(*arrays):
        rotate = np.random.random() > 0.5
        if rotate:
            angle = np.random.choice([1, 2, 3])
            arrays = [np.rot90(arr, k=angle) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        data2 = self.data2[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            if np.random.random() > 0.5:
                data, data2, label = self.flip(data, data2, label)
            else:
                data, data2, label = self.rotate(data, data2, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        data2 = np.asarray(np.copy(data2).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            data2 = data2[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        #     data2 = data2.unsqueeze(0)
        return data, data2, label
