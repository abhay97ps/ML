import xlrd
import matplotlib.pyplot as plt
import numpy as np

book = xlrd.open_workbook("/home/alpha/Labwork/ML/PA_2/Direct Classification data sets/Ionospheric Dataset 2/ionosphere.xls")
sheet = book.sheet_by_index(0)
cnt = 0
for i in range(0,351):
    if(sheet.cell_value(i,34) == 'g'):
        cnt +=1

print(cnt)
