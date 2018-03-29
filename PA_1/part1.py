import numpy as np
import matplotlib.pyplot as plt

def mean(array):
    mean = 0
    for i in range(len(array)):
        mean += array[i]
    mean = mean / (len(array)-1)
    return mean

def stdDev(array, mean):
    standDev = 0
    variance = 0
    for i in range(len(array)):
        variance += (array[i] - mean)**2
    variance = variance / (len(array)-1)
    standDev = np.sqrt(variance)
    return standDev

# 1-a: random points in range 1,2 with uniform distribution

random_values_uniform = np.random.uniform(1, 2, 1000)
count, bins, ignored = plt.hist(random_values_uniform, 100, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()

# calutaing mean and atandard deviation for 1-a

print("Mean for uniform distribution: " + str(mean(random_values_uniform)))
print("Standard Deviation for uniform distribution: " + str(stdDev(random_values_uniform, mean(random_values_uniform))))

# 1-b: random points with normal distribution

random_values_normal = np.random.normal(10, 5, 1000)
count, bins, ignored = plt.hist(random_values_normal, 100, normed=True)
plt.plot(bins, 1 / (5 * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - 10)**2 / (2 * 5**2)), linewidth=2, color='r')
plt.show()

# calculating mean and deviation for 1-b

print("Mean for normal distribution: " + str(mean(random_values_normal)))
print("Standard Deviation for normal distribution: " + str(stdDev(random_values_normal, mean(random_values_normal))))
