import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def plot(mu1, Sigma1, mu2, Sigma2, mu3, Sigma3):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = np.linspace(-20, 20, N)
    Y = np.linspace(-20, 20, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    A = multivariate_normal(mu1, Sigma1)
    B = multivariate_normal(mu2, Sigma2)
    C = multivariate_normal(mu3, Sigma3)
    Z1 = A.pdf(pos)
    Z2 = B.pdf(pos)
    Z3 = C.pdf(pos)
    # Create a surface plot and projected filled contour plot under it.
    Z = Z1 + Z2 + Z3
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

def contour(mu1, Sigma1, mu2, Sigma2, mu3, Sigma3,test):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-6, 18, N)
    Y = np.linspace(-12, 16, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    A = multivariate_normal(mu1, Sigma1)
    B = multivariate_normal(mu2, Sigma2)
    C = multivariate_normal(mu3, Sigma3)
    Z1 = A.pdf(pos)
    Z2 = B.pdf(pos)
    Z3 = C.pdf(pos)

    Z = Z1 + Z2 + Z3
    fig = plt.figure()
    plt.contour(X,Y,Z)
    plt.contour(X,Y,(Z1-Z2-Z3).reshape(X.shape),levels=[0])
    plt.contour(X,Y,(Z2-Z3-Z1).reshape(X.shape),levels=[0])
    plt.contour(X,Y,(Z3-Z1-Z2).reshape(X.shape),levels=[0])
    plt.scatter(test[0][0],test[0][1],color='r')
    plt.scatter(test[1][0],test[1][1],color='b')
    plt.scatter(test[2][0],test[2][1],color='g')
    ##############################################################
    w,v=np.linalg.eig(Sigma1)
    x1,y1=v
    w,v=np.linalg.eig(Sigma2)
    x2,y2=v
    w,v=np.linalg.eig(Sigma3)
    x3,y3=v

    plt.quiver(mu1[0],mu1[1],x1,y1)
    plt.quiver(mu2[0],mu2[1],x2,y2)
    plt.quiver(mu3[0],mu3[1],x3,y3)
    ###############################################################
    plt.show()