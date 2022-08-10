import torch

"""
Generator and Discriminator:
- The job of the generator is to spawn fake images that look like the training images
- the job of the discriminator is to look at an image and output whether or not it is a real training iamge
or a fake image from the data. 

Training:

- During training, the generator is constantly trying to fool the discriminator with better and better fake images,
while the discriminator is working to be a better detective

- Equilibrium is established when the discriminator is left to always guess at 50% confidence that the output generator
is real or fake. 
"""


# this is a test


