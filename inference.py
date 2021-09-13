# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division
import joblib
import os
from utils import convert_to_color_, convert_from_color_, get_device
from datasets import open_file
from model_utils import get_model, test
import numpy as np
import seaborn as sns
from skimage import io
import argparse
import torch

# Test options
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
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
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

group_test = parser.add_argument_group("Test")
group_test.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
group_test.add_argument(
    "--image_hsi",
    type=str,
    default=None,
    nargs="?",
    help="Path to an image on which to run inference.",
)
group_test.add_argument(
    "--image_lidar",
    type=str,
    default=None,
    nargs="?",
    help="Path to an image on which to run inference.",
)
group_test.add_argument(
    "--only_test",
    type=str,
    default=None,
    nargs="?",
    help="Choose the data on which to test the trained algorithm ",
)
group_test.add_argument(
    "--mat",
    type=str,
    default='HSI',
    nargs="?",
    help="In case of a .mat file, define the variable to call inside the file",
)
group_test.add_argument(
    "--mat_lidar",
    type=str,
    default='LiDAR',
    nargs="?",
    help="In case of a .mat file, define the variable to call inside the file",
)
group_test.add_argument(
    "--n_classes",
    type=int,
    default=None,
    nargs="?",
    help="When using a trained algorithm, specified  the number of classes of this algorithm",
)
# Training options
group_train = parser.add_argument_group("Model")
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
         "absent will be set by the model)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
MODEL = args.model
# Testing file
MAT = args.mat
MAT_LIDAR = args.mat_lidar
N_CLASSES = args.n_classes
INFERENCE = args.image_hsi
TEST_STRIDE = args.test_stride
CHECKPOINT = args.checkpoint

img_filename = os.path.basename(INFERENCE)
basename = MODEL + img_filename
dirname = os.path.dirname(INFERENCE)

img = open_file(INFERENCE)
if MAT is not None:
    img = img[MAT]
# Normalization
img = np.asarray(img, dtype="float32")
img = (img - np.min(img)) / (np.max(img) - np.min(img))


img_lidar = open_file(args.image_lidar)
if MAT_LIDAR is not None:
    img_lidar = img_lidar[MAT_LIDAR]
# Normalization
img_lidar = np.asarray(img_lidar, dtype="float32")
img_lidar = np.expand_dims(img_lidar, axis=2)
img_lidar = (img_lidar - np.min(img_lidar)) / (np.max(img_lidar) - np.min(img_lidar))

N_BANDS = (img.shape[-1], img_lidar.shape[-1])

hyperparams = vars(args)
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "device": CUDA_DEVICE,
        "ignored_labels": [0],
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", N_CLASSES)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


model, _, _, hyperparams = get_model(MODEL, **hyperparams)
model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))
probabilities = test(model, img, img_lidar, hyperparams)
prediction = np.argmax(probabilities, axis=-1)

filename = dirname + "/" + basename + ".tif"
io.imsave(filename, prediction)
basename = "color_" + basename
filename = dirname + "/" + basename + ".tif"
io.imsave(filename, convert_to_color(prediction))

# python inference.py --model S2ENet --checkpoint s2enet.pth --image_hsi D:/dataset/RS/multimodal/Houston2013/HSI.mat --image_lidar D:/dataset/RS/multimodal/Houston2013/LiDAR.mat  --n
# _classes 16
