from helper import *

path = '/home/alpha/Labwork/ML/PA_2/Linearly_seperable_data.txt'
data_file = open(path,'r')
X,Y = [],[]
for line in data_file:
    x,y = line.split()
    X.append(float(x))
    Y.append(float(y))

class1 = [X[0:500],Y[0:500]]
class2 = [X[500:1000],Y[500:1000]]
class3 = [X[1000:1500],Y[1000:1500]]
################################################################
c1_train = [class1[0][0:350],class1[1][0:350]]
c2_train = [class2[0][0:350],class2[1][0:350]]
c3_train = [class3[0][0:350],class3[1][0:350]]

c1_train_mean = [np.mean(c1_train[0]),np.mean(c1_train[1])]
c2_train_mean = [np.mean(c2_train[0]),np.mean(c2_train[1])]
c3_train_mean = [np.mean(c3_train[0]),np.mean(c3_train[1])]

c1_train_cov = np.cov(c1_train)
c2_train_cov = np.cov(c2_train)
c3_train_cov = np.cov(c3_train)

train_data = [[],[]]
train_data[0] = c1_train[0] + c2_train[0] + c3_train[0]
train_data[1] = c1_train[1] + c2_train[1] + c3_train[1]
common_cov = np.cov(train_data)
print(np.linalg.eigvals(common_cov))
#################################################################
c1_test = [class1[0][350:500],class1[1][350:500]]
c2_test = [class2[0][350:500],class2[1][350:500]]
c3_test = [class3[0][350:500],class3[1][350:500]]
test = [c1_test,c2_test,c3_test]
#################################################################
#case1
plot(c1_train_mean,common_cov,c2_train_mean,common_cov,c3_train_mean,common_cov)
contour(c1_train_mean,common_cov,c2_train_mean,common_cov,c3_train_mean,common_cov,test)
#case2
plot(c1_train_mean,c1_train_cov,c2_train_mean,c2_train_cov,c3_train_mean,c3_train_cov)
contour(c1_train_mean,c1_train_cov,c2_train_mean,c2_train_cov,c3_train_mean,c3_train_cov,test)
#case3 
plot(c1_train_mean,np.diag(np.diag(c1_train_cov)),c2_train_mean,np.diag(np.diag(c2_train_cov)),c3_train_mean,np.diag(np.diag(c3_train_cov)))
contour(c1_train_mean,np.diag(np.diag(c1_train_cov)),c2_train_mean,np.diag(np.diag(c2_train_cov)),c3_train_mean,np.diag(np.diag(c3_train_cov)),test)
