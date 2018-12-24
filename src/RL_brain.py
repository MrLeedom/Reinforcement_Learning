"""
有关强化学习中的Ｑ学习以及Sarsa的核心功能代码
"""
import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = action_space  #list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            #添加一个新的状态到q表中
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index = self.q_table.columns, name = state,)
            )
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        #行为选择
        if np.random.rand() < self.epsilon:
            #选择最好的行为
            state_action = self.q_table.loc[observation, :]
            #一些行为可能有同样的值，随机选择这些行为中的某一个
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            #选择随机行为
            action = np.random.choice(self.actions)
        return action
    def learn(self, *args):
        pass

#off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            #下一个状态不是终结状态的话
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()
        else:
            #下一个状态是终结状态
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)   #更新

#on-policy
class SarsaTable(RL):
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #下一个状态不是终结状态的话
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            #下一个状态是终结状态
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)   #更新