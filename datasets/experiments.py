import matplotlib
matplotlib.use('TkAgg')  # Necessary to run matplotlib

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import os

a = 'Animal/images/basset_hound_49.jpg'
a = 'Animal/images/Abyssinian_8.jpg'
b = a.replace('images', os.path.join('annotations', 'trimaps')).replace('jpg', 'png')

image = Image.open(a)
image = np.array(image)

# Load mask
mask = Image.open(b)
mask = np.array(mask)

plt.figure()
plt.imshow(image)
plt.figure()
plt.imshow(mask)
plt.show()