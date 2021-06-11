from utils import plot_image
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
OBECTIVE:
see how I can create the labels.csv for BBLox classification

TODO: create more image data and create corresponding labels

"""
# x_midpoint, y_midpoint, width, height
boxes = [(0.525, 0.5, 0.035, 0.9),
         (0.47, 0.52, 0.70, 0.05)]
image_1 = Image.open("./data/BBlox/BBloxImages/1.jpg")

plot_image(image_1, boxes)


