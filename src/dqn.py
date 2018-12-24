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