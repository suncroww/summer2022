import numpy as np
from random import randint
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# truth = np.ravel(np.loadtxt("Delorzo2.txt", delimiter=','))
# scatter = np.load("derenzo_50000_n_50.npy")[50:350,50:350,25]

# multiplier = 3

# x = truth[::3]*multiplier-142.5*multiplier
# y = truth[1::3]*multiplier-120.0*multiplier
# r = truth[2::3]*multiplier

size = 300

# xs = np.linspace(-10,10,size)
# ys = np.linspace(-10,10,size)
# xs,ys = np.meshgrid(xs,ys)

# data = np.zeros([size,size])

# def addcircle(data,x,y,r):
#     data = np.zeros([size,size])
#     for i in range(size):
#         for j in range(size):
#             if np.sqrt((i-(x+size/2))**2 + (j-(y+size/2))**2) < r/2:
#                 data[i][j] = 10
#     return data

# for i in range(0,len(x)):
#     data += addcircle(data, x[i], y[i], r[i])
#     print("Made circle at (" + str(x[i]) + ", " + str(y[i]) + ") with diameter " + str(r[i]))
# data = np.flipud(data)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# h = ax.pcolormesh(data, cmap='viridis')
# plt.axis('off')

# plt.savefig("drnztruth2.png",bbox_inches='tight',pad_inches=0)
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.axis("off")
ax1.set_aspect("equal")
ax2 = fig.add_subplot(122)
plt.axis("off")
ax2.set_aspect("equal")
plt.subplots_adjust(wspace=0.0,hspace=0.0)

# image = ImageOps.grayscale(Image.open('drnztruth2.png'))
# data = np.array(image).astype(float)
# np.save("data.npy", data)
data = np.load("data.npy")

for i in range(0,50):
    rot = randint(0,4)
    scatter = np.load("derenzo_50000_n_50.npy")[50:350,50:350,i-50]

    crop = (randint(0,100),randint(0,100),randint(200,300),randint(200,300))

    image = ImageOps.grayscale(Image.open('drnztruth2.png'))
    image = image.resize((size,size))
    image = image.crop(crop)
    image = image.resize((size,size))
    data = np.array(image).astype(float)
    # image.show()

    image = Image.fromarray(scatter)
    image = image.resize((size,size))
    image = image.crop(crop)
    image = image.resize((size,size))
    scatter = np.array(image).astype(float)
    # image.show()

    h = ax1.pcolormesh(np.rot90(data, rot))
    h = ax2.pcolormesh(np.rot90(scatter, rot))
    
    plt.savefig("ml/drnzbg_train/" + str(i) + ".png",bbox_inches='tight',pad_inches=0)
    print("Made training image #" + str(i))

# plt.colorbar(h)
# plt.show()