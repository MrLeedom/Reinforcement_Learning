import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../result/output/road.csv')
x = data['trips'].values
y = data['traveltime'].values

array_x = []
array_y = []
for i in range(200):
    array_x.append(5*i)
    mid = (int(y[2*i]) + int(y[2*i+1]) + int(y[2*i+2]) + int(y[2*i+3]) + int(y[2*i+4])) / 5
    array_y.append(mid)

plt.scatter(x,y,color='red',marker='s',alpha=0.5)
plt.plot(array_x,array_y,color='blue',marker='o',alpha=0.5)
plt.xlim((0,1000))
# plt.ylim((20,50))
plt.title('Travel of travel costs in Q-Learning')
plt.xlabel('Number of trips')
plt.ylabel('Travel time')
plt.savefig('../result/output/trips.jpg')
plt.show()