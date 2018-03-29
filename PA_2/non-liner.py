import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

path1 = '/home/alpha/Labwork/ML/PA_2/Non_Linearly_seperable_data_Class_1.txt'
path2 = '/home/alpha/Labwork/ML/PA_2/Non_Linearly_seperable_data_Class_2.txt'

data_file = open(path1,'r')
X,Y = [],[]
for line in data_file:
    x,y = line.split()
    X.append(float(x))
    Y.append(float(y))
class1 = [X,Y]

data_file = open(path2,'r')
X,Y = [],[]
for line in data_file:
    x,y = line.split()
    X.append(float(x))
    Y.append(float(y))
class2 = [X,Y]


##################################################
class1Train = [class1[0][0:700],class1[1][0:700]]
class2Train = [class2[0][0:700],class2[1][0:700]]
class1Test = [class1[0][700:1000],class1[1][700:1000]]
class2Test = [class2[0][700:1000],class2[1][700:1000]]
##################################################

c1TrainMean = [np.mean(class1Train[0]),np.mean(class1Train[1])]
c1TrainCov = np.cov(class1)
c2TrainMean = [np.mean(class2Train[0]),np.mean(class2Train[1])]
c2TrainCov = np.cov(class2)
#########################################
#########################################

N = 100
X = np.linspace(10, 40, N)
Y = np.linspace(10, 40, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
A = multivariate_normal(c1TrainMean, c1TrainCov)
B = multivariate_normal(c2TrainMean, c2TrainCov)
Z1 = A.pdf(pos)
Z2 = B.pdf(pos)
Z = Z1 + Z2
###################################################################
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1,
                antialiased=True, cmap=cm.viridis)
#cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
# Adjust the limits, ticks and view angle
ax.set_zlim(0, 0.1)
ax.set_zticks(np.linspace(0, 0.1, 10))
ax.view_init(27, -21)
plt.show()
###################################################################

#contour
fig = plt.figure()
plt.contour(X,Y,Z)
plt.contour(X,Y,(Z1-Z2).reshape(X.shape),levels=[0])
##############################################################
w,v=np.linalg.eig(c1TrainCov)
x1,y1=v
w,v=np.linalg.eig(c2TrainCov)
x2,y2=v
#eigenvector
plt.quiver(c1TrainMean[0],c1TrainMean[1],x1,y1,color='r')
plt.quiver(c2TrainMean[0],c2TrainMean[1],x2,y2)
###############################################################
plt.scatter(class1Test[0],class1Test[1],color='r')
plt.scatter(class2Test[0],class2Test[1],color='b')
    
###############################################################
plt.show()
