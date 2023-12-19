import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("seaborn")


x = np.linspace(0,1,100)
noise = np.random.normal(loc=0, scale=0.25,size=100)
y= np.sin(x*1.5*np.pi)
y_noise = y +noise

plt.figure(figsize=(10,6))
plt.plot(x,y,color = 'darkblue', label = 'f(x)')
plt.scatter(x,y_noise,facecolors = 'none',
           edgecolor = 'darkblue',
           label='f(x) + noise')


plt.legend()
plt.title('Noisy sine function and bell shaped kernel at x=0.2')
plt.show()


from math import ceil
import numpy as np
from scipy import linalg

def kernel_function(xi,x0,tau=0.5):
    return np.exp(-(xi-x0)**2/(2*tau**2))

def lwr(x,y,xtest,tau = 0.005):
    yest = np.zeros(len(xtest))
    
    for i in range(len(xtest)):
        weights = np.array([np.exp(-(x-xtest[i])**2/(2*tau**2))])
        
        b = np.array([np.sum(weights*y),np.sum(weights*y*x)])
        A=np.array([[np.sum(weights),np.sum(weights*x)],
                   [np.sum(weights*x),np.sum(weights*x*x)]])
        theta = linalg.solve(A,b)
        #invA=linalg.inv(A)
        #theta=invA.dot(b)
        yest[i] = theta[0] + theta[1] * xtest[i]
        
    return yest


tau = 0.3 #CHANGE THIS VALUE TO 0.001, 0.005, 0.1, 0.3, 0.8 and take ss of plot
xtest = np.linspace(0,1,50)
yest_bell = lwr(x,y,xtest,tau)
print(len(xtest),len(yest_bell))


plt.figure(figsize=(10,6))
plt.scatter(x,y_noise,facecolors='none',
           edgecolor = 'darkblue', label = 'sin() + noise')
plt.fill(x,kernel_function(x,0.2,tau),color='lime',
        alpha = 0.5, label = 'Kernel')
plt.scatter(xtest,yest_bell,color='red',
           label = 'Loess: bell shaped kernel')
plt.legend()
plt.title('Sine with noise: Loess regression and bell shaped kernel')
plt.show()


ytest_actual = np.sin(xtest*1.5*np.pi)
from sklearn import metrics
print("Accuracy = ",metrics.r2_score(yest_bell,ytest_actual))
