
import numpy as np
import math
from scipy import special
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt

def bessel_integrand(m,x,theta): #theta in radians
	return np.cos(m*theta - x*np.sin(theta))
'''
def rect_integral(f,a,b,n):
    h = float(b-a)/n
    s = 0.
    s += f(m,x,a)/2.
    for i in range(1, n):
        s += f(m,x,a + i*h)
    s += f(m,x,b)/2.
    return s * h

def Bessel(m,x):
	return rect_integral(bessel_integrand,0,np.pi,1000)/np.pi
	
#To compare to scipy Bessel function:
plt.figure(figsize=(10,5))
plt.title('Plot of numerically integrated Bessel functions and scipy Bessel functions')
xarray = np.arange(0,20,0.1)
marray = [1,2,3]
for m in marray:
	plt.plot(xarray,[Bessel(m,x) for x in xarray],'o',label='Numerical m = %s' % m)
	plt.plot(xarray,special.jv(m,xarray),label='Scipy m = %s' % m)
plt.xlabel('x')
plt.ylabel('Bessel (order m)')
plt.ylim([-0.5,1])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3)
plt.show()
'''

#To plot PSF:
def psf(xarray,yarray,a,R,l,I0):
	#xarray and yarray are arrays
	xv,yv = np.meshgrid(xarray,yarray)
	q_grid = np.sqrt(xv**2. + yv**2.)
	x_arg_grid = (2.*np.pi*a*q_grid)/(l*R)
	image = I0*((2.*special.jv(1,x_arg_grid))/x_arg_grid)**2.
	return image
'''
dxy = 0.01
xarray = np.arange(-2,2+dxy,dxy)
yarray = np.arange(-2,2+dxy,dxy)
a = 2.
R = 20.
l = 10e-7
I0 = 10e13
image = psf(xarray,yarray,a,R,l,I0)
image[len(xarray)/2,len(yarray)/2] = 4.*I0

plt.imshow(image, clim =[0,3], cmap ='Blues', extent=[-2,2,-2,2])
plt.title('Point spread function (PSF)')
plt.xlabel('cm')
plt.ylabel('cm')
plt.colorbar()
plt.show()
'''

#To convolve an astronomy image with a PSF:

with open('m31.bmp', 'r') as f:
    data = bytearray(f.read())

galaxy = np.array(data).reshape(3,len(data)/3)
galaxy = np.dstack((galaxy[0],galaxy[1],galaxy[2]))[:,18:].reshape((590,736,3))
galaxy = galaxy[:,:,0]+galaxy[:,:,1]+galaxy[:,:,2]


#galaxy = misc.imread('m31.jpeg', flatten=0)

plt.imshow(galaxy)
plt.title('Galaxy Image')
plt.colorbar()
plt.show()

xarray = np.linspace(-736/2,736/2,736)
yarray = np.linspace(-590/2,590/2,590)
a = 2.
R = 20.
l = 10e-7
I0 = 10e13
galaxy_psf = psf(xarray,yarray,a,R,l,I0)

galaxy_convolved = signal.convolve2d(galaxy,galaxy_psf)

plt.imshow(galaxy_convolved)
plt.title('Galaxy convolved with PSF')
plt.colorbar()
plt.show()



