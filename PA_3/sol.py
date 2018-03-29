from helper import *

path = '/home/alpha/Labwork/ML/PA_3/Direct Classification data sets/Iris Dataset 3/irisdata.xls'
book = xlrd.open_workbook(path)
sheet  = book.sheet_by_index(0)

# reading and storing the data
data = []
for i in range(0,150):
    temp = [float(sheet.cell_value(i,0)),float(sheet.cell_value(i,1)),float(sheet.cell_value(i,2)),float(sheet.cell_value(i,3)),sheet.cell_value(i,4)]
    data.append(temp)

# grouping the data points in groups (5 groups) 
group = [[],[],[],[],[]]
for i in range(0,5):
    group[i] = data[0+i*10:10+i*10] + data[50+i*10:60+i*10] + data[100+i*10:110+i*10] 
all_points = group[0]+group[1]+group[2]+group[3]+group[4]

# the process is repeated 5 times with test size 1/5th of data
ConfMat = np.array([[0,0,0],[0,0,0],[0,0,0]])
Acc = 0
for i in range(0,5):
    test = group[i]
    train = [point for point in all_points if point not in test]
    lik_clas,post_class = classify(train,test)
    acc_lik, cm_lik = crossValidate(lik_clas)
    ConfMat += np.array(cm_lik)
    Acc += acc_lik
ConfMat = ConfMat/5
Acc = Acc/5

print('Confusion Matrix')
print(ConfMat)
print('Accuracy: ' + str(Acc))
print('-------------------------')