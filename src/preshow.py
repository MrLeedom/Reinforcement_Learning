import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

###################将每条边的旅行时间特点进行显示##########################
data = pd.read_csv('../data/edge.csv')
link = {}
x_num = []
for i in range(1,13):
    x_num.append(i)
    array_x = []
    array_y = []
    current = data[data['edgeId'] == i]
    for j in range(current.iloc[:,0].size):
        array_x.append(j+1)
        array_y.append(current.iloc[j,2])
    link[i] = array_y
    plt.figure()
    plt.scatter(array_x,array_y,alpha=0.6)
    plt.xlabel('Time Range(1-24) /5min')
    plt.ylabel('Average travel-time on each time-range(mins)')
    plt.xlim((1,24))
    plt.ylim((3,11)) 
    plt.savefig('../result/origin/'+str(i)+'.jpg')
    # plt.show()
    plt.close()

##################将１２条边用箱型图的形式展示出来##########################
boxdata = []
for k in range(len(link)):
    boxdata.append(link[k+1])
plt.figure()
plt.boxplot(boxdata,positions=np.array(x_num))
plt.title('Overall Situation')
plt.xlabel('Links')
plt.ylabel('Travel Time Distribution')
plt.xticks(x_num)
plt.savefig('../result/origin/overall.jpg')
plt.show()
print('运行成功!!!')
