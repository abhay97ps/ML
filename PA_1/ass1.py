import numpy as np
import matplotlib.pyplot as plt

# 1-a: random points in range 1,2 with nform distribution

random_values_uniform = np.random.uniform(1, 2, 1000)
count, bins, ignored = plt.hist(random_values_uniform,100, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()


# 1-b: random points with normal distribution

random_values_normal = np.random.normal(10, 5, 1000)
count, bins, ignored = plt.hist(random_values_normal,100, normed=True)
plt.plot(bins, 1 / (5 * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - 10)**2 / (2 * 5**2)), linewidth=2, color='r')
plt.show()


# second part of calculating mean and deviation
mean = 0
for i in range(1000):
    mean += random_values_normal[i]
mean = mean / 1000
print("mean: " + str(mean))

standDev = 0
variance = 0
for i in range(1000):
    variance += (random_values_normal[i] - mean)**2
variance = variance / 1000
standDev = np.sqrt(variance)
print("standDev: " + str(standDev))

##############################################################################
# 3
mean = [0, 0]
cov = [0, 0, 0, 0]
cov[0] = [[4, 0], [0, 2]]
cov[1] = [[2, 0], [0, 2]]
cov[2] = [[4, 1], [1, 2]]
cov[3] = [[4, -1], [-1, 2]]

for i in range(4):
    x, y = np.random.multivariate_normal(mean, cov[i], 10000).T
    plt.plot(x, y, 'x')
    plt.show()
    mean_x = 0
    mean_y = 0
    for i in range(10000):
        mean_x += x[i]
        mean_y += y[i]
    mean_cal = [mean_x / 10000, mean_y / 10000]
    print("mean: " + str(mean_cal))
    temp1, temp2, temp3 = 0, 0, 0
    for i in range(10000):
        temp1 += (x[i] - mean_cal[0])**2
    for i in range(10000):
        temp2 += (x[i] - mean_cal[0])*(y[i] - mean_cal[1])
    for i in range(10000):
        temp3 += (y[i] - mean_cal[1])**2
    cov_calculated = [[temp1/10000, temp2/10000],
                      [temp2/10000, temp3/10000]]
    print("cov matrix:" + str(cov_calculated))
