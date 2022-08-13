import numpy as np
from scipy.special import jv
from scipy.special import yn
from scipy.special import spherical_jn
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as Axes3D
from scipy import optimize
from scipy import signal
from scipy import interpolate
from math import atan2

# 2D

# x = np.linspace(-20,20,1000)

# for i in range(0,5):
#     J = jv(i,x)
#     plt.plot(x,J,label = r'$J_' + str(i) + '(x)$')

# 3D

x = np.linspace(-10,10,220)
y = np.linspace(-10,10,220)
x, y = np.meshgrid(x, y)

r = np.sqrt(x**2 + y**2)

def angle(x,y):
    theta = atan2(y,x)
    return theta

z = 0
z = spherical_jn(9,r)
# print(z)
# print(np.ravel(z))



def func(order):
    z=0
    for i in range(0,order):
        z += spherical_jn(i,r)
    return np.array(np.ravel(z))

def besselbg(p, image, rows, cols, order):
    x = p[0]
    y = p[1]
    r = p[2]
    ddx = np.linspace(-x/r, (rows-x)/r, rows)
    ddy = np.linspace(-y/r, (cols-y)/r, cols)
    xv, yv = np.meshgrid(ddx, ddy) # makes an x and y matrix of vales?
    r = np.sqrt(xv**2 + yv**2)
    bessel = spherical_jn(order, r)
    eval = 0
    # plt.imshow(bessel)
    # plt.show()
    plt.plot(r,bessel)
    return np.reshape(image - eval, rows*cols); 

current_image = np.load("McDisplayDerenzo.npy") 
order = 15
img = current_image[110]
rows, cols = img.shape
# #gives the rows and columns of the image we are looking for 
r_0 = cols/2. if rows > cols else rows/2.
# #defines the radius to create a unit circle
x_0 = rows/2.; # x 0 = half width
y_0 = cols /2.; # x 0 = half height
p0 = [ x_0 , y_0 , r_0 ]
# p1 , success =\
# zop = optimize.leastsq(besselbg, p0, args=(image, rows, cols, order))

# derenzo = plt.imshow(img)
# plt.show()
# bessel = plt.imshow(spherical_jn(2,r)*.0006)
# plt.show()
# besselify = plt.imshow(img - spherical_jn(2,r)*.0006)
# plt.show()

def getimgvalue(image,x,y,rows):
    return np.ravel(image)[rows*y+x]

# fig = plt.figure("sphericalBessel")
# ax = fig.add_subplot(111)
# h = ax.pcolormesh(x, y, z, cmap="plasma")
# h = ax.pcolormesh(z,cmap="plasma")
# plt.colorbar(h)
# plt.show()

# plt.plot(r,z)
# plt.show()

def plotimg():
    image = np.load("McDisplayDerenzo.npy")
    # rows = 220
    # ix=0 ; iy=0
    # ix, iy = np.meshgrid(ix, iy)
    # ir = np.sqrt(ix**2+iy**2)
    # plt.plot(ir,getimgvalue(image,ix,iy,rows))
    plt.plot(range(0,220), image[0])

# copied from stackoverflow...
# def radial_profile(data, center):
#     y, x = np.indices((data.shape))
#     r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
#     r = r.astype(np.int)

#     tbin = np.bincount(r.ravel(), data.ravel())
#     nr = np.bincount(r.ravel())
#     radialprofile = tbin / nr
#     return radialprofile 

# center, radi = (110, 110), 55
# rad = radial_profile(img, center)

# plt.plot(rad[radi:])

data = [2,3,6,8,3,2,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,4,3,4,7,5,3,2,4,6,8,8,7,5,12,3,3,4,5,8,7,6,4,2]

def y(theta,x):
    return theta[0]*x+theta[1]

xs = np.linspace(0,6)
a0 = 1 ; b0 = 0

def fun(theta):
    return data - y(theta,xs)

theta0 = [a0,b0]
res = optimize.least_squares(fun,theta0)

# plt.plot(data)
# plt.plot(y(res.x, xs))
# plt.show()



# from scipy.special import yn
# from scipy.optimize import curve_fit
# from numpy import sin,linspace,pi

# a=1#choose the order a here!
# func = lambda var,b : yn(a,b*var)

# x=linspace(1,2*pi,100)
# y=sin(x)/2.#we fit this as an example
# [b], pcov = curve_fit(func, x, y) # x and y are my data points
# print(b)

# plt.plot(y,color="blue")
# plt.plot(yn(a,x),color="orange")
# plt.show()

# copy pasted 3d leastsq example
def h(theta, x, y):
    return theta[2] * (x - theta[0])**2 + theta[3] * (y - theta[1])**2

xs = np.linspace(-1, 1, 100)
ys = np.linspace(-1, 1, 100)
gridx, gridy = np.meshgrid(xs, ys)
x0 = 0.1; y0 = -0.15; a = 1; b = 2; noise = 0.5
hs = h([x0, y0, a, b], gridx, gridy)
hs += noise * np.random.default_rng().random(hs.shape)

def fun(theta):
    return (h(theta, gridx, gridy) - hs).flatten()

theta0 = [0, 0, 1, 2]
res3 = optimize.least_squares(fun, theta0)

fig2 = plt.figure("3d")
ax2 = fig2.add_subplot(111)
# hh = ax.plot_surface(xs,ys,hs,cmap="plasma")
hh = ax2.pcolormesh(hs,cmap="plasma")
plt.colorbar(hh)
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
hs = h(res3.x, gridx, gridy)
hhh = ax3.pcolormesh(hs,cmap="plasma")
plt.colorbar(hhh)
plt.show()

# #save functioning 2d fitter here
# def bessel(theta,x):
#     return theta[0]*jv(x, theta[1])
# coeff0 = 1 ; order0 = 1

# def fun(theta):
#     return data - bessel(theta, xs)

# theta0 = [coeff0, order0]
# res = optimize.least_squares(fun, theta0)
# print(res.x)

# plt.plot(data)
# plt.plot(bessel(res.x,xs))
# plt.show()

# #attempt at iteration
# def bessel(theta,order, x):
#     return theta[0]*jv(x, order)
# coeff0 = 1 ; order0 = 1

# def fun(theta):
#     return data - bessel(theta, order, xs)

# fit = []
# fit.append(bessel([0],0,xs))
# print(fit)

# order = 5
# theta0 = [coeff0]
# for n in range(0,order):
#     res = optimize.least_squares(fun, theta0)
#     print(res.x)
#     fit += bessel(res.x,order,xs)

# plt.plot(data)
# print(fit)
# plt.plot(fit, color="red")
# plt.show()

size = 220
z = spherical_jn(1,r)

aoi = np.ravel(z)
avgintensity = np.mean(aoi)

# INTENSITY FILTER
# maybe make it not a strict cutoff? gradated? that doesnt make sense.
def hipass(image):
    aoi = np.ravel(image)
    avgintensity = np.mean(aoi)
    index = 0
    for i in aoi:
        if i > avgintensity:
            aoi[index] = 100
        index += 1
    return aoi

# INTERPOLATION
# please amend this to like, actually interpolate things. rn its scuffed
def interpolate(aoi):
    index = 0
    for i in aoi:
        if i == 100 and index!=0 and index!=(len(aoi)-1):
            # aoi[index] = (aoi[index-1] + aoi[index+1])/2
            aoi[index] = avgintensity
        elif index == 0: aoi[index] = aoi[1]
        elif index == (len(aoi)-1): aoi[index] = aoi[len(aoi)-2]
        index += 1
    return aoi

z = hipass(z)
z = interpolate(z)
z = np.reshape(z, (size,size))