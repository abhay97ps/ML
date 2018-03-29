import numpy as np

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

m1 = 0.0
std1 = 1.0
m2 = 1.0
std2 = 1.0

result = solve(m1,m2,std1,std2)
print(result)