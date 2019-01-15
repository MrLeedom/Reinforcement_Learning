#-*- coding:utf-8 -*-
'''
   @author:leedom

   Created on Mon Jan 14 21:55:14 2019
   description:测试一下几个强化学习效率问题
'''
import numpy as np
import pandas as pd
import random
import datetime
import pickle
from sklearn.utils import shuffle
import RL_brain
#迷宫矩阵
maze = np.array([
    [0,0,0,0,0,0,],
    [1,0,1,1,1,1,],
    [1,0,1,0,0,0,],
    [1,0,0,0,1,1],
    [0,1,0,0,0,0,]
])
#起点
state_state = (0, 0)
#终点
target_state = (2,5)
#要保存的q_table的文件路径
q_learning_table_path = './config/q_learning_table.pkl'
def get_next_state_reward(current_state, action, reward_dict): 
    """
    :param current_state: 当前状态
    :param action: 动作
    :return: next_state下个状态，reward奖励值，done游戏是否结束
    """ 
    done = False 
    if action == 'up': 
        next_state = (current_state[0] - 1, current_state[1]) 
    elif action == 'down': 
        next_state = (current_state[0] + 1, current_state[1]) 
    elif action == 'left':
        next_state = (current_state[0], current_state[1] - 1) 
    else: 
        next_state = (current_state[0], current_state[1] + 1) 
    if next_state[0] < 0 or next_state[0] >= maze.shape[0] or next_state[1] < 0 or next_state[1] >= maze.shape[1] or maze[next_state[0], next_state[1]] == 1: 
        # 如果出界或者遇到1，保持原地不动 
        next_state = current_state 
        reward = reward_dict.get('reward_0') 
        # 此处done=True,可理解为进入陷阱，游戏结束，done=False，可理解为在原地白走一步，受到了一次惩罚，但游戏还未结束 # done = True 
    elif next_state == target_state: # 到达目标 
        reward = reward_dict.get('reward_2') 
        done = True 
    else: # maze[next_state[0],next_state[1]] == 0 
        reward = reward_dict.get('reward_1') 
    return next_state, reward, done
def train1():
    reward_dict = {'reward_0':-1,'reward_1':0.1,'reward_2':1}
    actions = ('up', 'down', 'left', 'right')
    q_learning_table = RL_brain.QLearningTable(actions)
    #迭代次数
    iterators = 500
    start = datetime.datetime.now()
    for _ in range(iterators):
        current_state = state_state
        print(_)
        while True:
            q_learning_table.check_state_exist(str(current_state))
            action = q_learning_table.choose_action(str(current_state))
            next_state,reward,done = get_next_state_reward(current_state, action, reward_dict)
            q_learning_table.learn(str(current_state),action,reward,str(next_state))
            if done:
                break
            current_state = next_state
    print('game over')
    end = datetime.datetime.now()
    time = (end - start).microseconds
    print('time:%dus'%(time))
    #保存对象模型(也就是一个矩阵)到q_learning_table_path
    with open(q_learning_table_path,'wb') as pkl_file:
        pickle.dump(q_learning_table, pkl_file)
def train2():
    reward_dict = {'reward_0':-1,'reward_1':0.1,'reward_2':1}
    actions = ('up', 'down', 'left', 'right')
    q_learning_table = RL_brain.SarsaTable(actions)
    #迭代次数
    iterators = 500
    
    start = datetime.datetime.now()
    for _ in range(iterators):
        current_state = state_state
        action = q_learning_table.choose_action(str(current_state))
        while True:
            q_learning_table.check_state_exist(str(current_state))
            action_ = q_learning_table.choose_action(str(current_state))
            next_state,reward,done = get_next_state_reward(current_state, action, reward_dict)
            q_learning_table.learn(str(current_state),action,reward,str(next_state),action_)
            if done:
                break
            current_state = next_state
            action = action_
    print('game over')
    end = datetime.datetime.now()
    time = (end - start).microseconds
    print('time:%dus'%(time))
    #保存对象模型(也就是一个矩阵)到q_learning_table_path
    with open(q_learning_table_path,'wb') as pkl_file:
        pickle.dump(q_learning_table, pkl_file)
def train3():
    reward_dict = {'reward_0':-1,'reward_1':0.1,'reward_2':1}
    actions = ('up', 'down', 'left', 'right')
    q_learning_table = RL_brain.SarsaLamdaTable(actions)
    #迭代次数
    iterators = 500
    
    start = datetime.datetime.now()
    for _ in range(iterators):
        current_state = state_state
        action = q_learning_table.choose_action(str(current_state))
        while True:
            q_learning_table.check_state_exist(str(current_state))
            action_ = q_learning_table.choose_action(str(current_state))
            next_state,reward,done = get_next_state_reward(current_state, action, reward_dict)
            q_learning_table.learn(str(current_state),action,reward,str(next_state),action_)
            if done:
                break
            current_state = next_state
            action = action_
    print('game over')
    end = datetime.datetime.now()
    time = (end - start).microseconds
    print('time:%dus'%(time))
    #保存对象模型(也就是一个矩阵)到q_learning_table_path
    with open(q_learning_table_path,'wb') as pkl_file:
        pickle.dump(q_learning_table, pkl_file)
if __name__ == "__main__":
    train1()
    train2()
    train3()

