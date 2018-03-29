import numpy as np
import xlrd
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

# function for estimation and classification
def classify(train,test):        
    c = [[],[],[]]
    for i in range(0,len(train)):
        if train[i][4]=="Iris-setosa":
            c[0].append(train[i][0:4])
        elif train[i][4]=="Iris-versicolor":
            c[1].append(train[i][0:4])
        elif train[i][4]=="Iris-virginica":
            c[2].append(train[i][0:4])
    # caculating prior of classes
    c1_cnt = len(c[0])
    c2_cnt = len(c[1])
    c3_cnt = len(c[2])
    total = c1_cnt + c2_cnt + c3_cnt
    prior_c1 = c1_cnt/total
    prior_c2 = c2_cnt/total
    prior_c3 = c3_cnt/total

    # Calculating mu and sigma using ML estimate

    mu1 = np.mean(c[0],axis=0)
    sigma1 = np.cov(c[0],rowvar=False)

    mu2 = np.mean(c[1],axis=0)
    sigma2 = np.cov(c[1],rowvar=False)
    
    mu3 = np.mean(c[2],axis=0)
    sigma3 = np.cov(c[2],rowvar=False)

    # Getting the likelihood gaussian

    C1_likli = multivariate_normal(mu1,sigma1)
    C2_likli = multivariate_normal(mu2,sigma2)
    C3_likli = multivariate_normal(mu3,sigma3)

    # Getting the posterior

    def C1_post(temp):
        return (C1_likli.pdf(temp))*prior_c1
    def C2_post(temp):
        return (C2_likli.pdf(temp))*prior_c2
    def C3_post(temp):
        return (C3_likli.pdf(temp))*prior_c3

    # Classification using Likelihoods
    class_classified_L = [[],[],[],[]] #class 1,23 and unclassified(for those on decision boundary)
    for i in range(0,len(test)):
        point = test[i][0:4]
        argmin = np.max([C1_likli.pdf(point),C2_likli.pdf(point),C3_likli.pdf(point)])
        flag = [-1,-1] # to check for unclassified classes
        if C1_likli.pdf(point) == argmin:
            flag[0] = 1
            flag[1] +=1
        if C2_likli.pdf(point) == argmin:
            flag[0] = 2
            flag[1] +=1
        if C3_likli.pdf(point) == argmin:
            flag[0] = 3
            flag[1] +=1

        if flag[1] == 0:
            class_classified_L[flag[0]-1].append(test[i])
        elif flag[1] > 0:
            class_classified_L[3].append(test[i])
    
    # Classification using posterior
    class_classified_P = [[],[],[],[]] #class 1,23 and unclassified(for those on decision boundary)
    for i in range(0,len(test)):
        point = test[i][0:4]
        argmin = np.max([C1_post(point),C2_post(point),C3_post(point)])
        flag = [-1,-1] # to check for unclassified classes
        if C1_post(point) == argmin:
            flag[0] = 1
            flag[1] +=1
        if C2_post(point) == argmin:
            flag[0] = 2
            flag[1] +=1
        if C3_post(point) == argmin:
            flag[0] = 3
            flag[1] +=1

        if flag[1] == 0:
            class_classified_P[flag[0]-1].append(test[i])
        elif flag[1] > 0:
            class_classified_P[3].append(test[i])
    
    return class_classified_L,class_classified_P

def indexer(text):
    if text=="Iris-setosa":
        return 0
    elif text=="Iris-versicolor":
        return 1
    elif text=="Iris-virginica":
        return 2
    return -1

def crossValidate(classified_points):
    ConfMatrix = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(0,len(classified_points[0])):
        ConfMatrix[indexer(classified_points[0][i][4])][0]+=1
    for i in range(0,len(classified_points[1])):
        ConfMatrix[indexer(classified_points[1][i][4])][1]+=1
    for i in range(0,len(classified_points[2])):
        ConfMatrix[indexer(classified_points[2][i][4])][2]+=1

    total = np.sum(ConfMatrix)
    TP = ConfMatrix[0][0] + ConfMatrix[1][1] + ConfMatrix[2][2]
    return (TP/total)*100, ConfMatrix
