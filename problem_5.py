import numpy as np
import matplotlib.pyplot as plt

def der(f,x,delta):
	return (f(x+delta)-f(x))/delta

def f(x):
	return x*(x-1)
	
x = np.linspace(0,2,100)

#test to see below if derivative is a straight line:
f_der = der(f,x,0.0001)
plt.figure(1)
plt.title('Plot of function and its derivative')
plt.plot(x,f_der,'bo', label='numerical derivative')
plt.plot(x,(2*x-1),'g-', label= 'Analytical derivative 2x-1')
plt.plot(x,f(x),'b-', label= 'original function x(x-1)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3)

#plot for different deltas;
deltas= np.logspace(-14,-4,10)
f_der_deltas=der(f,1,deltas)
plt.figure(2)
ax = plt.subplot(111)
plt.title('Plot of numerical derivative at x=1 for a range of deltas')
plt.plot(deltas,f_der_deltas,'o')
plt.plot(deltas,f_der_deltas)
plt.axhline(y=1)
plt.xlabel('delta')
plt.ylabel('$f_{der}(1)$')
ax.set_xscale('log')
#plt.ylim([1.0,1.0002])
plt.show()


