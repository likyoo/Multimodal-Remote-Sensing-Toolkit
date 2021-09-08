# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR MULTIMODAL REMOTE SENSING.
(https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit)

This script allows the user to run several deep models
against various multimodal datasets. It is developed on
the top of DeepHyperX (https://github.com/nshaud/DeepHyperX)

"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io

# Visualization
import seaborn as sns
import visdom
import random

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_lidar_data,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
    padding_image,
    restore_from_padding,
    seed_torch,
)
from datasets import get_dataset, MultiModalX, open_file, DATASETS_CONFIG
from model_utils import get_model, train, test, save_model
import argparse

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "EndNet, "
    "Early_fusion_CNN, "
    "Middle_fusion_CNN, "
    "Late_fusion_CNN, "
    "Cross_fusion_CNN, "
    "FusAtNet, "
    "S2ENet, "
    "Others, ",
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Set random seed",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--train_val_split",
    type=float,
    default=1,
    help="Percentage of samples to use for training and validation, "
         "'1' means all training data are used to train",
)
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=10,
    help="Percentage of samples to use for training (default: 10%%) and testing",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
SAMPLE_TRAIN_VALID = args.train_val_split
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

# set random seed
seed_torch(seed=args.seed)

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img1, img2, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)


if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)

def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Show the image and the ground truth
display_dataset(img1, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
display_lidar_data(img2, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img1, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = (img1.shape[-1], img2.shape[-1])

# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)['TRLabel']
        test_gt = open_file(TEST_GT)['TSLabel']
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    # delete
    # display_predictions(convert_to_color(open_file('../Houston2013/gt.mat')['gt']), viz, caption="ground truth")

    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams["weights"] = torch.from_numpy(weights)
    # Neural network
    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

    # Split train set in train/val
    if SAMPLE_TRAIN_VALID != 1:
        train_gt, val_gt = sample_gt(train_gt, SAMPLE_TRAIN_VALID, mode="random")
    else:
        # Use all training data to train the model
        _, val_gt = sample_gt(train_gt, 0.95, mode="random")

    # Generate the dataset
    train_dataset = MultiModalX(img1, img2, train_gt, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        # pin_memory=hyperparams['device'],
        shuffle=True,
    )
    val_dataset = MultiModalX(img1, img2, val_gt, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        # pin_memory=hyperparams['device'],
        batch_size=hyperparams["batch_size"],
    )

    print(hyperparams)
    print("Network :")
    with torch.no_grad():
        for input, input2,  _ in train_loader:
            break
        summary(model.to(hyperparams["device"]), [input.size()[1:], input2.size()[1:]])
        # We would like to use device=hyperparams['device'] altough we have
        # to wait for torchsummary to be fixed first.

    if CHECKPOINT is not None:
        model.load_state_dict(torch.load(CHECKPOINT))

    try:
        train(
            model,
            optimizer,
            loss,
            train_loader,
            hyperparams["epoch"],
            scheduler=hyperparams["scheduler"],
            device=hyperparams["device"],
            supervision=hyperparams["supervision"],
            val_loader=val_loader,
            display=viz,
        )
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass

    probabilities = test(model, img1, img2, hyperparams)

    try:
        prediction = np.argmax(probabilities, axis=-1)
        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )
    except:
        probabilities = restore_from_padding(probabilities, patch_size=[hyperparams["patch_size"], hyperparams["patch_size"]])
        prediction = np.argmax(probabilities, axis=-1)
        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        gt=convert_to_color(test_gt),
        caption="Prediction vs. test ground truth",
    )

    flt_test_gt = test_gt.reshape(-1)
    flt_prediction = prediction.reshape(-1)

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
