import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# data = np.load("filtering/3_dots_500k.npy")[100:300,100:300,100]
# data = np.load("Truth_e1.npy")
# data = np.load("derenzo_50000_n_50.npy")[50:350,50:350,25]
# data = np.load("filtering/McDisplayDerenzo.npy")[110]

# image = ImageOps.grayscale(Image.open('brainslices/40.jpg'))
# image = image.crop((59,59,211,211))

# plt.imshow(image)

data = np.load("brain.npy")[10]
# data = np.load("derenzo_30000_vac_bg_50.npy")[10:390,10:390,40]

plt.imshow(data, cmap='viridis')
plt.axis('off')
# plt.savefig("xcatslice.png", bbox_inches='tight', pad_inches=0)
plt.show()