#You might want to upload McDisplayDerenzo.py if it isn’t there. 

import numpy as np
from math import atan2
from numpy import cos , sin , conjugate , sqrt
from zernike import RZern
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import ( AutoMinorLocator , MultipleLocator )


def kepler_slicer (volume, slice_plane, slice_value):
    # first step is to define a new 2d plane that can be graphed.
    # if the slice_plane is ”x” then the x values are held 
    # constant and a new array of y and z is created.
    # Symmetric for all axis. x>yz , y>xz , z>xy 
    if slice_plane == "x" or slice_plane == "X" :
        use_slice = volume [slice_value]
    if slice_plane == "y" or slice_plane == "Y" : 
        use_slice = volume [: , slice_value]
    if slice_plane == "z" or slice_plane == "Z" : 
       use_slice = volume[: , : , slice_value] 
    return use_slice

# shows what the zernike error from the image is ? 
def zernike_error (p , image, rows, cols , order):
    cart = RZern(order); # something from the Zernike polynomial libray ,
    # defines an object? Polynomial up to order?
    x = p [0] #position x?
    y = p [1] #position y?
    r = p [2] #position radius? unit circle radius?
    # makes a set of positions from left to right and top to bottom to check 
    # same thing done for y
    ddx = np.linspace(-x/r , (rows-x)/r, rows)
    # makes a range of numbers of size rows from left to right?
    ddy = np.linspace(-y/r , (cols-y)/r, cols)
    xv, yv = np.meshgrid(ddx, ddy) # makes an x and y matrix of vales?
    cart.make_cart_grid (xv, yv , unit_circle=False )
    eval = cart.eval_grid(cart.fit_cart_grid(image)[0] , matrix = True); 
    # evaluation of Zernike Polynomials somehow 
    # print(np.size(eval))
    return np.reshape(image - eval, rows*cols); 


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # load the image that is going to be fit 
    # current_image = np.load("derenzo_600.npy") 
    current_image = np.load("3_dots_500k.npy")[100:300,100:300,100]
    order = 15

    # img = kepler_slicer(current_image , "x" , 110) 
    img = np.load("3_dots_500k.npy")[100:300,100:300,100]
    # makes the 2d matrix that will be fit
    # img = test image

    rows, cols = img.shape
    #gives the rows and columns of the image we are looking for 
    r_0 = cols/2. if rows > cols else rows/2.
    #defines the radius to create a unit circle
    x_0 = rows/2.; # x 0 = half width
    y_0 = cols /2.; # x 0 = half height
    p0 = [ x_0 , y_0 , r_0 ]
    p1 , success =\
    optimize.leastsq(zernike_error , p0 , args=(img , rows , cols , order)) 
    cart = RZern(order); 

    x = p1 [ 0 ]
    y = p1 [ 1 ] 
    r = p1 [ 2 ] 
    print ("Best Estimate coordinate origin : x="\
    +str(x)+" ,y="+str(y)+" , r="+str(r))
    ddx = np.linspace(-x/r, (rows-x)/r, rows)
    ddy = np.linspace(-y/r, (cols-y)/r, cols)
    xv, yv = np.meshgrid(ddx , ddy)
    cart.make_cart_grid(xv , yv , unit_circle=False )
    eval = cart.eval_grid(cart.fit_cart_grid(img)[0] , matrix = True); 
    #print (np.size(eval))
    #plot zernike
    a_plot = plt.imshow(eval, cmap='hot')
    plt.colorbar (a_plot, orientation='vertical')
    plt.show()

    #plot original graph
    # test = kepler_slicer(current_image, "x", 110)
    test = np.load("3_dots_500k.npy")[100:300,100:300,100]
    b_plot = plt.imshow(test, cmap='hot')
    plt.colorbar(b_plot, orientation='vertical') 
    plt.show()

    #plot fitted graph
    fitted_slice = test - eval 
    #  fitted slice = test image − eval
    print(fitted_slice[81][125])
    print(fitted_slice[109][109])
    c_plot = plt.imshow(fitted_slice, cmap='hot')
    # plt.axis("off")
    # plt.savefig("help.png",bbox_inches='tight',pad_inches=0)
    plt.colorbar (c_plot, orientation='vertical')
    #plt.grid(color=”w”)
    plt.show()