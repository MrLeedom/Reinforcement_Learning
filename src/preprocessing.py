import numpy as np
import random

###################随机数的生成#########################
edge = {}
for k in range(1,13):
    edge[k] = []
    for i in range(1,25):
        edge[k].append(random.randint(4,10))

##################产生的数据存放至文件###################
fileObject = open('../data/edge.csv','w')
firstLine = 'edgeId,timesId,travelTime'
fileObject.write(firstLine)
fileObject.write('\n')
for m in range(len(edge)):  
    for item in range(len(edge[m+1])):
        print(str(m+1)+','+str(item+1)+','+str(edge[m+1][item]),end = '\n', file = fileObject)