import numpy as np
import math
import scipy.special
import matplotlib.pyplot as plt

def z_series(x,y,n):
	#this function takes in 2D arrays of x and y and a total iteration number n
	c = x + 1j*y
	#first slice of z_all (z_0) is all 0
	z_all = np.zeros((n,len(x),len(x[0])),dtype=complex) #3D array
	for i in range(1,n):
		z_all[i] = z_all[i-1]**2. + c
	return z_all

dxy = 0.005
xarray = np.arange(-2,2+dxy,dxy)
yarray = np.arange(-2,2+dxy,dxy)
xv,yv = np.meshgrid(xarray,yarray)

#To plot an image with values of 1 and 0 for converging and diverging points, respectively:
N = 100
array = abs(np.nan_to_num(z_series(xv,yv,N)[-1]))
array[array > 0] = 1

plt.imshow(array, cmap ='Blues', extent=[-2,2,-2,2])
plt.title('Mandelbrot Set')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

#To plot an image with values equal to the iteration number for divergence:
array_3D = z_series(xv,yv,N) #entire 3D output array
array_index = np.zeros(np.shape(array))
for i in range(len(xarray)):
	for j in range(len(yarray)):
		diverge_index = np.where(array_3D[:,i,j] > 10e100)[0]
		array_index[i,j] = diverge_index[0] if len(diverge_index) > 0 else N

plt.imshow(array_index, cmap ='Blues', extent=[-2,2,-2,2])
plt.title('Mandelbrot Set (divergence iteration # up to N=%s)' % N)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()

#To zoom in to a region of the set and provide more resolution to see self-similarity:
#(Plotting 1 and 0 for convergence and divergence)
dxy_fine = 0.001
xarray_zoom = np.arange(-1.75,-0.75+dxy_fine,dxy_fine)
yarray_zoom = np.arange(-0.5,0.5+dxy_fine,dxy_fine)
xv_zoom,yv_zoom = np.meshgrid(xarray_zoom,yarray_zoom)

array_zoom = abs(np.nan_to_num(z_series(xv_zoom,yv_zoom,N)[-1]))
array_zoom[array_zoom > 0] = 1

plt.imshow(array_zoom, cmap ='Blues', extent=[-1.75,-0.75,-0.5,0.5])
plt.title('Mandelbrot Set (zoomed in, more resolution)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
