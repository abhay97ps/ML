import numpy as np
import matplotlib.pyplot as plt


def mean(x, y):
    mean_x = 0
    mean_y = 0
    for i in range(len(x)):
        mean_x += x[i]
        mean_y += y[i]
    mean_cal = [mean_x / len(x), mean_y / len(x)]
    return mean_cal


def covMatrix(x, y, mean):
    temp1, temp2, temp3 = 0, 0, 0
    for i in range(len(x)):
        temp1 += (x[i] - mean[0])**2
    for i in range(len(x)):
        temp2 += (x[i] - mean[0]) * (y[i] - mean[1])
    for i in range(len(y)):
        temp3 += (y[i] - mean[1])**2
    cov_calculated = [[temp1 / (len(x) - 1), temp2 / (len(x) - 1)],
                      [temp2 / (len(x) - 1), temp3 / (len(x) - 1)]]
    return cov_calculated


mean_given    = [0, 0] 
cov = [[[4, 0], [0, 2]], [[2, 0], [0, 2]],
       [[4, 1], [1, 2]], [[4, -1], [-1, 2]]]

for i in range(4):
    x, y = np.random.multivariate_normal(mean_given, cov[i], 10000).T
    plt.plot(x, y, 'x')
    plt.show()

    # mean vector and covariance matrix

    print("Mean Vector: " + str(mean(x, y)))
    print("Covariance Matrix:" + str(covMatrix(x, y, mean(x, y))))
