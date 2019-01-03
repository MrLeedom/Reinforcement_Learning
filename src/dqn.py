###########################################
'''
强化学习是学习一个最优策略，可以让智能体在特定环境中，根据当前状态，做出行动，从而获得最大回报
DQN要做的就是将卷积神经网络ＣＮＮ和Ｑ-learning结合起来，CNN的输入是原始图像数据（如何将非图像数据视作其输入数据是难点？)
输出则是每个动作的价值评估价值函数(Q值)
'''
###########################################
import numpy as np
import pandas as pd
import os
from collections import deque
from sklearn.utils import shuffle
from keras.losses import mean_squared_error
import copy
import random
from keras.models import Model,load_model
from keras.layers import Input,Dense,Reshape,Conv2D,Flatten

#迷宫矩阵
maze = np.array(
    [[0,0,0,0,0,0,],
    [1,0,1,1,1,1,],
    [1,0,1,0,0,0,],
    [1,0,0,0,1,1,],
    [0,1,0,0,0,0,]]
)
#要存储的模型文件
model_name = '../data/dqn_model.h5'
#当走到(row,col)时，令迷宫矩阵在(row,col)处的值为POS_VALUE
TMP_VALUE = 2
#起点
start_state_pos = (0,0)
#终点
target_state_pos = (2,5)
#动作字典
actions = dict(
    up = 0,
    down = 1,
    left = 2,
    right = 3
)
#动作维度，也是神经网络要输出的维度
action_dimension = len(actions)
#奖励值字典，到达终点奖励１，走０奖励-0.01,走１或出界奖励-1
reward_dict = {'reward_0':-1,'reward_1':-0.01,'reward_2':1}

#将迷宫矩阵转为图片格式的ｓｈａｐｅ(height,width,channel)
def matrix_to_img(row,col):
    state = copy.deepcopy(maze)
    state[row,col] = TMP_VALUE
    #维度转换
    state = np.reshape(state,newshape=(1,state.shape[0],state.shape[1],1))
    # print(state)
    return state


class DQNAgent:
    def __init__(self,agent_model=None):
        self.memory = deque(maxlen=100)
        self.alpha = 0.01
        self.gamma = 0.9   #decay rate
        #动作的探索率exploration
        self.epsilon = 1
        #探索率的最小值
        self.epsilon_min = 0.2
        #探索衰减率
        self.epsilon_decay = 0.995
        #学习率
        self.learning_rate = 0.001
        if agent_model is None:
            self.model = self.dqn_model()
        else:
            self.model = agent_model
    
    #模型
    def dqn_model(self):
        inputs = Input(shape =(maze.shape[0],maze.shape[1],1))
        layer1 = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same')(inputs)
        layer2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(layer1)
        layer3 = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same')(layer2)
        layer4 = Flatten()(layer3)
        predictions = Dense(action_dimension,activation='softmax')(layer4)
        model = Model(inputs=inputs,outputs=predictions)
        model.compile(optimizer='sgd',loss=mean_squared_error,)
        return model

    #保存当前状态current_state,动作action,奖励值reward,下个状态next_state,游戏是否结束done
    def remember(self,current_state, action, reward, next_state, done): 
        self.memory.append((current_state, action, reward, next_state, done)) 

    # 选择动作，self.epsilon为动作探索阈值 
    def choose_action(self, state): # 随机选择动作 
        if np.random.rand() < self.epsilon: 
            action = random.choice(list(actions.keys())) 
            action = actions.get(action) 
            return action 
            # 根据当前状态预测要选择的动作 
        else: 
            act_values = self.model.predict(state) 
            # 因为pd.Series数据的最大值可能出现多个，而argmax()只取第一个，故使用sklearn中的shuffle将其打乱顺序， 
            action = np.argmax(shuffle(pd.Series(act_values[0]))) 
            return action

    # 从记忆容器self.memory中随机选择(current_state, action, reward, next_state, done),然后送入模型进行训练 
    def repay(self, batch_size): 
        batch_size = min(batch_size, len(self.memory)) 
        #从记忆池中取出batch_size个目标
        batch_random_choice = np.random.choice(len(self.memory),batch_size) 
        for i in batch_random_choice: 
            current_state, action, reward, next_state, done = self.memory[i] 
            # target_f 目标值 
            target_f = self.model.predict(current_state) 
            if done: 
                target = reward 
            else: 
                target = reward + self.alpha * (self.gamma * np.max(self.model.predict(next_state)[0]) - target_f[0][action]) 
            target_f[0][action] = target 
            # 训练模型，更新权重 
            self.model.fit(current_state, target_f, nb_epoch=2, verbose=0) 

            # 更新探索率 
            if self.epsilon > self.epsilon_min: 
                self.epsilon = self.epsilon * self.epsilon_decay 
            else: 
                self.epsilon = self.epsilon_min
# 环境 
class Environ: 
    def __init__(self): 
        pass 

    # 根据当前状态current_state和动作action，返回next_state, reward, done 
    def step(self,current_state, action): 
        # 定位当前状态的索引 
        row, col = np.argwhere(current_state == TMP_VALUE)[0,1:3] 
        done = False 
        if action == actions.get('up'): 
            next_state_pos = (row - 1, col) 
        elif action == actions.get('down'): 
            next_state_pos = (row + 1, col) 
        elif action == actions.get('left'): 
            next_state_pos = (row, col - 1) 
        else: 
            next_state_pos = (row, col + 1) 
        if next_state_pos[0] < 0 or next_state_pos[0] >= maze.shape[0] or next_state_pos[1] < 0 or next_state_pos[1] >= maze.shape[1]  or maze[next_state_pos[0], next_state_pos[1]] == 1: 
            # 如果出界或者遇到1，保持原地不动 
            next_state = copy.deepcopy(current_state) 
            reward = reward_dict.get('reward_0') 
            # 此处done=True,可理解为进入陷阱，游戏结束，done=False，可理解为在原地白走一步，受到了一次惩罚，但游戏还未结束 
            # done = True 
        elif next_state_pos == target_state_pos: 
            # 到达目标 
            next_state = matrix_to_img(target_state_pos[0],target_state_pos[1]) 
            reward = reward_dict.get('reward_2') 
            done = True 
        else: # maze[next_state[0],next_state[1]] == 0 
            next_state = matrix_to_img(next_state_pos[0], next_state_pos[1]) 
            reward = reward_dict.get('reward_1') 
        return next_state, reward, done

def train(): 
    # 如果模型已存在，加载模型 
    if os.path.exists(model_name): 
        agent_model = load_model(model_name) 
        agent = DQNAgent(agent_model=agent_model) 
    else: 
        agent = DQNAgent() 
        # 环境 
        environ = Environ() 
        # 迭代次数 
        episodes = 10000 
        for e in range(episodes): 
            # 在每次游戏开始时复位状态参数 
            current_state = matrix_to_img(start_state_pos[0],start_state_pos[1]) 
            i = 0 
            while(True): 
                i = i + 1 
                # 选择行为 
                action = agent.choose_action(current_state) 
                # 在环境中施加行为推动游戏进行 
                next_state, reward, done= environ.step(current_state,action) 
                # 记忆先前的状态，行为，奖励值与下一个状态 
                agent.remember(current_state, action, reward, next_state, done) 
                if done: 
                    # 游戏结束，跳出循环，进入下次迭代 
                    print("episode: {}, step used:{}" .format(e, i)) 
                    break 
                # 使下一个状态成为下一帧的新状态 
                current_state = copy.deepcopy(next_state) 
                # 通过之前的经验训练模型 
                if i % 100 == 0: 
                    agent.repay(100) 
                # 每迭代2000次，保存一次模型 
                if (e+1) % 1000 == 0: 
                    agent.model.save(model_name)

def predict(): 
    # actions 键值对互换 
    actions_new = dict(zip(actions.values(),actions.keys())) 
    # 加载模型 
    agent_model = load_model(model_name) 
    environ = Environ() 
    current_state = matrix_to_img(start_state_pos[0], start_state_pos[1]) 
    # 最多走100步，超过100游戏结束 
    for i in range(100): 
        # 选择行为，action预测结果示例[[0.0686022  0.0237738  0.05400459 0.85361934]] 
        action = agent_model.predict(current_state) 
        # action最大值的索引 即为要执行的下一个动作 
        action = np.argmax(action[0]) 
        # 在环境中施加行为推动游戏进行 
        next_state, reward, done = environ.step(current_state, action) 
        #np.argwhere得到又是二维数组[[0,2,3,0],[0,1,1,0]]
        print('current_state: {}, action: {}, next_state: {}'.format(np.argwhere(current_state==TMP_VALUE)[0,1:3], actions_new[action], np.argwhere(next_state==TMP_VALUE)[0,1:3])) 
        # 如果游戏结束，跳出循环 
        if done: 
            break 
        # 使下一个状态成为下一帧的新状态 
        current_state = next_state

if __name__ == 
"__main__":
    # train()
    predict()
