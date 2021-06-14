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
# boxes = [(0.38, 0.74, 0.28, 0.32),
#          (0.47, 0.52, 0.70, 0.05)]
boxes = [(0.42, 0.66, 0.2, 0.27),
         (0.64, 0.67, 0.22, 0.28),
         (0.63, 0.37, 0.177, 0.23)    ]
image_1 = Image.open("./data/BBlox/BBloxImages/3.jpg")

plot_image(image_1, boxes)


