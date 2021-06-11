from Vision.utils import plot_image
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
OBECTIVE:
see how I can create the labels.csv for BBLox classification

- Not sure how to get the by, bx, bw, bh for boudning boxes of elements
- How to handle multiple classifications on single image?

"""
# x_midpoint, width, y_midpoint, height
boxes = [(0.5, 0.05, 0.5, 0.9),
         (0.5, 0.9, 0.5, 0.05)]
image_1 = Image.open("./data/BBlox/BBloxImages/1.jpg")

plot_image(image_1, boxes)


"""
image_1 = Image.open("./data/BBlox/BBloxImages/1.jpg")
#image_1 = np.array(image_1)

image_1 = image_1.resize((480, 480))

fig = plt.figure(figsize=(20, 15))
plt.grid(False)
plt.axis('off')

plt.imshow(image_1)
"""
