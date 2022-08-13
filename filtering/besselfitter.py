import numpy as np
from scipy.special import jv
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
from scipy import optimize
from PIL import Image, ImageOps
import math
import cmath

# IMPORT IMAGE AND EXTRACT DATA
def init(image):
    print("deal with this later")

#temporary setup
size = 220

xs = np.linspace(-10,10,size)
ys = np.linspace(-10,10,size)
xs, ys = np.meshgrid(xs, ys)
rs = np.sqrt(xs**2 + ys**2)
phis = np.linspace(0,2*math.pi,size)

data = np.load("McDisplayDerenzo.npy")[110]
# image = ImageOps.grayscale(Image.open('scanme.png')).transpose(Image.FLIP_TOP_BOTTOM)
# data = np.array(image).astype(float)

order = 7

theta0 = np.zeros(2*(order+1))

# def bessel(theta,x,y,order):
#     return theta*spherical_jn(order, rs)

# def cosine(theta,x,y,freq):
#     return theta*np.cos(freq*phis)

# # plt.plot(phis,np.cos(phis))
# # plt.show()

# def fun(theta):
#     func = data*0
#     func += data
#     for i in range(order+1):
#         func -= bessel(theta[2*i],xs,ys,i)*cosine(theta[2*i+1],xs,ys,i)
#     return np.ravel(func)

# res = optimize.least_squares(fun, theta0)
# print(res.x)

# fig = plt.figure()
# ax = fig.add_subplot(221)
# h = ax.pcolormesh(data, cmap="plasma")
# plt.colorbar(h)

# ax = fig.add_subplot(222)
# hs = spherical_jn(0,rs)*0
# for i in range(order+1):
#     hs += spherical_jn(i,rs)*res.x[2*i]*np.cos(i*phis)*res.x[2*i+1]
# h = ax.pcolormesh(hs, cmap="plasma")
# plt.colorbar(h)

# ax = fig.add_subplot(223)
# # ax = fig.add_subplot(111)
# h = ax.pcolormesh(data - hs, cmap="plasma")
# plt.colorbar(h)

# ax = fig.add_subplot(224)
# plt.plot(range(2*(order+1)), res.x, marker='o', markerfacecolor='red', markersize=4)

# ax.set_aspect("equal")
# plt.axis("off")
# plt.savefig("noisyderenzo"+str(order)+".png",bbox_inches='tight',pad_inches=0)
# np.save("values.npy", data-hs)


fig = plt.figure()
ax = fig.add_subplot(111)

# z = spherical_jn(0,rs)*0
# z = jv(0,phis)*0
# for i in range(0,10):
#     z = jv(i,phis)*np.cos(i*phis)
#     plt.plot(z)
# h = ax.pcolormesh(z,cmap='plasma')
# plt.colorbar(h)

xtest = np.linspace(-10,10,100)
ytest = np.linspace(-10,10,100)
rtest = np.sqrt(xtest**2 + ytest**2)
phitest = np.arctan(ytest/xtest)

h = ax.pcolormesh(rtest,np.cos(phitest),cmap='plasma')

plt.show()
