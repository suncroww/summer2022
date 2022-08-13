import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

size = 100

xs = np.linspace(-10,10,size)
ys = np.linspace(-10,10,size)
xs,ys = np.meshgrid(xs,ys)

def addcircle(data,x,y):
    r = np.random.randint(5,20)
    data = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            if np.sqrt((i-x)**2 + (j-y)**2) < r:
                data[i][j] = 2
            elif np.sqrt((i-x)**2 + (j-y)**2) < r+1:
                data[i][j] = 1.7
            elif np.sqrt((i-x)**2 + (j-y)**2) < r+2:
                data[i][j] = 1.3
    return data

def addnoise(data):
    data += np.random.default_rng().random(data.shape)
    return data

data = np.zeros([size,size])

def generate(data):
    data = np.zeros([size,size])
    for i in range(6):
        data += addcircle(data,np.random.randint(10,size-10),np.random.randint(10,size-10))
    return data

fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.axis("off")
ax1.set_aspect("equal")
ax2 = fig.add_subplot(122)
plt.axis("off")
ax2.set_aspect("equal")
plt.subplots_adjust(wspace=0.0,hspace=0.0)

for i in range(1,101):
    data = generate(data)
    h = ax1.pcolormesh(data,cmap='plasma')
    data += np.random.default_rng().random(data.shape)
    h = ax2.pcolormesh(data,cmap='plasma')
    plt.savefig("noisegeometries/train/t" + str(i) + ".png", bbox_inches='tight', pad_inches=0, dpi=200)