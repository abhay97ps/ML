import xlrd
import matplotlib.pyplot as plt
import numpy as np

book = xlrd.open_workbook("/home/alpha/Labwork/ML/PA_2/Direct Classification data sets/Iris Dataset 3/irisdata.xls")
sheet = book.sheet_by_index(0)

class1 = [[],[]]
class2 = [[],[]]
class3 = [[],[]]

for i in range(0,150):
    if(sheet.cell_value(i,4)=="Iris-setosa"):
        class1[0].append(sheet.cell_value(i,0))
        class1[1].append(sheet.cell_value(i,1))
    if(sheet.cell_value(i,4)=="Iris-versicolor"):
        class2[0].append(sheet.cell_value(i,0))
        class2[1].append(sheet.cell_value(i,1))
    if(sheet.cell_value(i,4)=="Iris-virginica"):
        class3[0].append(sheet.cell_value(i,0))
        class3[1].append(sheet.cell_value(i,1))

plt.hist(class1[0],bins = 'auto', histtype='step' ,normed=True,color = 'r')
#plt.show()
plt.hist(class2[0],bins = 'auto', histtype='step' ,normed=True,color = 'g')
#plt.show()
plt.hist(class3[0],bins = 'auto',histtype='step' , normed=True,color = 'b')
plt.show()

plt.hist2d(class1[0],class1[1],normed=True)
cbar = plt.colorbar();
cbar.ax.set_ylabel('Counts')


plt.show()
plt.hist2d(class2[0],class2[1],normed=True)
cbar = plt.colorbar();
cbar.ax.set_ylabel('Counts')


plt.show()
plt.hist2d(class3[0],class3[1],normed=True)
cbar = plt.colorbar();
cbar.ax.set_ylabel('Counts')

plt.show()
