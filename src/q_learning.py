import tensorflow as tf
import numpy as np
import random
import math
import pickle
import pandas as pd
from sklearn.utils import shuffle

#路网模型,9*9矩阵，其中１表示可走，０表示无法直接相连
road_network = np.array([
    [0,1,0,1,0,0,0,0,0],
    [1,0,1,0,1,0,0,0,0],
    [0,1,0,0,0,1,0,0,0],
    [1,0,0,0,1,0,1,0,0],
    [0,1,0,1,0,1,0,1,0],
    [0,0,1,0,1,0,0,0,1],
    [0,0,0,1,0,0,0,1,0],
    [0,0,0,0,1,0,1,0,1],
    [0,0,0,0,0,1,0,1,0]
])

###################路网的边的性质###########################
data = pd.read_csv('../data/edge.csv')
micro_network = {}
for i in range(1,13):
    insert_data = {}
    current = data[data['edgeId'] == i]
    array1 = current.iloc[:,1].values
    array2 = current.iloc[:,2].values
    for m in range(len(array1)):
        insert_data[array1[m]] = array2[m]
    micro_network[i] = insert_data

###################路网节点的性质############################
node_data = pd.read_csv('../data/constraint.csv')
#内部结构  1:{'pre_node':0,'direction':2,'edge':10,'next_node':2},'orientation':1|2|3|4
node = {}
for k in range(1,10):
    current = node_data[node_data['id'] == k]
    array = current.iloc[:,1].values
    if k == 9:
        node[9] = {}
        node[9]['orientation'] = (current.iloc[:,2].values)[0]
        break
    node[k] = {}
    node[k]['orientation'] = (current.iloc[:,2].values)[0]
    current_data = array[0].split('|')
    for h in range(len(current_data)):
        node[k][h] = {}
        item = current_data[h]
        aim = item.split('-')
        node[k][h]['pre_node'] = aim[0]
        node[k][h]['direction'] = aim[1]
        node[k][h]['edge'] = aim[2]
        node[k][h]['next_node'] = aim[3]
#起点
start_state = 1
#终点
target_state = 9
#要保存的ｑ_table的文件路径
q_learning_table_path = '../result/output/q_learning_table.pkl'
########################Q表的定义##############################
class QLeaningTable:
    def __init__(self, alpha=0.01, gamma=0.9):
        #self.alpha,self.gamma是Ｑ函数中需要用到的两个参数
        self.alpha = alpha
        self.gamma = gamma
        self.traveltime = 0
        self.minTravelTime = 120
        #奖励（惩罚）值
        self.reward_dict = {'reward_0':-1,'reward_1':0.1,'reward_2':10,'reward_3':20}
        #动作
        self.actions = ('0-0','0-1','0-2','1-0','1-1','1-2','2-0','2-1','2-2','3-0','3-1','3-2')
        self.q_table = pd.DataFrame(columns=self.actions)

    def get_next_state_Reward(self, first_state, current_state, action,traveltime):
        """
        :param  current_state:当前状态
        :param  action:动作
        :return next_state下一个状态，reward奖赏值,done游戏是否结束
        """
        done = False
        flag = True
        next_state = str(current_state)
        #判断该动作来自的方位是哪里
        orient = action.split('-')[0]
        #判断前一个节点是否在当前节点的某方位中
        positions = node[current_state]['orientation'].split('|')

        key = -1
        #记录下前个节点是在当前节点的什么方位
        if int(current_state) != 1:
            for i in range(len(positions)):
                if str(first_state) == positions[i]:
                    key = i
                    break

            if key == int(orient):
                for item in node[current_state]:
                    if item == 'orientation':
                        continue
                    if node[current_state][item]['pre_node'] == str(first_state) and node[current_state][item]['direction'] == action.split('-')[1]:
                        next_state = node[current_state][item]['next_node']
                        things = node[int(next_state)]['orientation'].split('|')
                        for k in range(len(things)):
                            if things[k] == str(current_state):
                                prefix = str(k)
                                break
                        number = math.floor(traveltime/5)
                        if number == 0:
                            number += 1
                        if number > 24:
                            flag = False
                            break
                        self.traveltime = traveltime + int(micro_network[int(node[current_state][item]['edge'])][number])
                        break
        else:
            #只更新1-0,1-1,1-2
            # if orient == '1':
            for item in node[current_state]:
                if item == 'orientation':
                    continue
                # print(node[current_state][item])
                # print(action.split('-')[1])
                # print(node[current_state][item]['pre_node'])
                if node[current_state][item]['pre_node'] == str(first_state) and node[current_state][item]['direction'] == action.split('-')[1]:
                    next_state = node[current_state][item]['next_node']
                    things = node[int(next_state)]['orientation'].split('|')
                    for k in range(len(things)):
                        if things[k] == str(current_state):
                            prefix = str(k)
                            break
                    number = math.floor(traveltime/5)
                    if number == 0:
                        number += 1
                    if number > 24:
                        flag = False
                        break
                    self.traveltime = traveltime + int(micro_network[int(node[current_state][item]['edge'])][number])
                    break

        if str(next_state) == str(current_state):  #在该点并未有移动,也就是说需要重新换动作来做
            reward = self.reward_dict.get('reward_0')#此时的当前状态为前个状态，下个状态为当前状态，相当于未变化
            next_state = current_state
            current_state = first_state
            prefix = orient
            
        elif next_state == str(target_state): #到达目标
            reward = self.reward_dict.get('reward_2')
            done = True
            if self.traveltime < self.minTravelTime:
                self.minTravelTime = self.traveltime
                reward = self.reward_dict.get('reward_3')
        else:
            reward = self.reward_dict.get('reward_1')
            
        return str(current_state),next_state, reward, done, flag,prefix

    #根据返回的reward和next_state更新q_table
    def learn(self, current_state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_sa = self.q_table.loc[next_state, action]
        #这一块需要取到的最大值实际上跟动作的前方位来确定的，而不是一味地在其中取最大值
        orient = action.split('-')[0]
        if orient == 0:
            start = 0
            end = 2
        elif orient == 1:
            start = 3
            end = 5
        elif orient == 2:
            start = 6
            end = 8
        else:
            start = 9
            end = 11
        max_next_q_sa = self.q_table.ix[next_state,start:end].max()
        #套用公式：Ｑ函数
        new_q_sa = q_sa + self.alpha * (reward + self.gamma*max_next_q_sa - q_sa)
        #更新q_table
        self.q_table.loc[current_state,action] = new_q_sa

    #如果state不在q_table中，便在q_table中添加该state
    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = pd.Series(np.zeros(len(self.actions)), index = self.actions)
    
    #执行动作的探索和利用
    def choose_action(self, prefix, state, random_num = 0.8):
        pool = [0,1,2]
        if prefix == '0':
            start = 0
            end = 3
        elif prefix == '1':
            start = 3
            end =6
        elif prefix == '2':
            start = 6
            end =9
        else:
            start = 9
            end = 12
        series = pd.Series(self.q_table.ix[state,start:end])
        #以0.8的概率执行action，尝试更多的可能性．总是做最好的选择，意味着你可能会错过一些从未探索的道路
        #为了避免这种情况，可以添加一个随机项，而未必总是选择对当前来说最好的action
        if random.random() > random_num:
            action = random.choice(pool)
            action = prefix + '-' + str(action)
        else:
            #因为pd.Series数据的最大值可能出现多个，而argmax()只取第一个，故使用sklearn中的shuffle将其打乱顺序
            #随机选取最大值的索引，选取最大值的action有利于q_table快速收敛
            ss = shuffle(series)
            action = ss.argmax()
        return action

#训练
def train():
    fileObject = open('../result/output/road2.csv','a+')
    print('trips,route,traveltime',end='\n',file=fileObject)
    q_learning_table = QLeaningTable()
    #迭代次数
    iterate_num = 2000
    for _ in range(iterate_num):
        #每次迭代都是从start_state开始
        current_state = start_state
        pre_state = 0
        tripTime = 0
        prefix = '1'
        q_learning_table.traveltime = 0
        print('第{}次迭代'.format(_+1))
        trip = []
        trip.append(start_state)
        while True:
            #先检查current_state是否已在q_table中，注意将current_state以字符串的形式存到q_table中
            q_learning_table.check_state_exist(str(current_state))
            #获取当前状态的执行动作
            action = q_learning_table.choose_action(prefix, str(current_state))
            #根据当前状态current_state和动作action,获取下一个状态next_state,奖励值reward以及游戏是否结束done
            pre_state,next_state, reward, done, flag,prefix = q_learning_table.get_next_state_Reward(pre_state, current_state,action,tripTime)
            #开始学习，更新q_table
            q_learning_table.learn(str(pre_state),action,reward,str(next_state))
            #current_state调到下个状态
            tripTime = q_learning_table.traveltime
            current_state = int(next_state)
            if current_state != trip[-1]:
                trip.append(current_state)
            #如果游戏结束，跳出while循环，进入下次迭代
            if done or flag == False:
                break
        ###############打印轨迹和与之对应的旅行时间################
        print('trip{}:{},time:{}min'.format(_+1,trip,tripTime))
        ##################将我们的执行动作保存到文件中###############
        route = ''
        for item in trip:
            if item == target_state:
                route += str(item)
                break
            route += str(item) + '->' 
        print(str(_+1)+','+route+','+str(tripTime),end='\n',file = fileObject)
    fileObject.close()
            
               
    print('game over')
    #保存对象q_learning_table 到文件q_learning_table_path
    with open(q_learning_table_path, 'wb') as pkl_file:
        pickle.dump(q_learning_table,pkl_file)   

######################用预测模型来测量准确性###################################
def predict():
    #读取ｑ_table
    with open(q_learning_table_path, 'rb') as pkl_file:
        q_learning_table = pickle.load(pkl_file)
    print('start_state:{}'.format(start_state))
    pre_state = 0
    tripTime = 0
    prefix = '1'
    current_state = start_state
    step = 0
    # print(q_learning_table.traveltime)
    # print(q_learning_table.q_table)
    while  True:
        step = step + 1
        action = q_learning_table.choose_action(prefix, str(current_state),random_num = 1)
        #预测阶段，reward用不到了，故使用_代替
        pre_state,next_state, reward, done, flag,prefix = q_learning_table.get_next_state_Reward(pre_state, current_state,action,tripTime)
        #输出动作和下个状态
        print('step:{step},state:{state}'.format(step=step,state=next_state))

        #如果done或者步数超过１００，游戏结束退出
        if done or flag == False or step > 100:
            if int(next_state) == target_state:
                print('success')
                tripTime = q_learning_table.traveltime
            else:
                print('fail')
            break
        #跳转到下个状态
        else:
            tripTime = q_learning_table.traveltime
            current_state = int(next_state)
    print(tripTime)
if __name__ == '__main__':
    train()
    predict()