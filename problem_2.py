import numpy as np
import math
import matplotlib.pyplot as plt

def binomial(n,k):
	if k == 0:
		return 1
	else:
		return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

#Pascal's triangle for first 20 lines
def pascal(n_total):
	for n in range(n_total):
		line = [str(binomial(n,k)) for k in range(n+1)]
		print ' '.join(line)


#Coin tossing:
def atleastk(k_lower,n,p):
	return sum([binomial(n,k)*(p**k)*(1.-p)**(n-k) for k in range(k_lower,n+1)])


#Simulating coin tossing for N = 10,100,1000:
N_all = range(10,1000)
fraction_all = []

p = 0.2
for N in N_all:
	total = sum(np.random.random(N) < p)
	fraction_all.append(float(total)/N)

plt.figure(figsize=(10,5))
plt.title('Fraction of heads as function of N tosses for p = %s' % p)
plt.plot(N_all,fraction_all,'o')
plt.axhline(y=p)
plt.xlabel('N')
plt.ylabel('Fraction of heads')
plt.show()
