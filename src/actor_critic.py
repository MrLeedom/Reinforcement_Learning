"""
Actor-Critic演员评判家，使用TD-error作为标准条件
"""
import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  #保证生成的随机数每次一致

OUTPUT_GRAPH = True
MAX_EPISODE = 3000
#如果一次训练中的总奖赏reward > 200便重新加载环境
DISPLAY_REWARD_THRESHOLD = 200  
#一次训练中最多的要经历的时间步数
MAX_EP_STEPS = 1000
RENDER = False   #是否重新加载一次环境
#TD error的奖赏折扣
GAMMA = 0.9
#ACTOR的学习率
LR_A = 0.001
#critic的学习率
LR_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped
#状态和行为的相关定义
N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1,n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  #TD-error

        with tf.variable_scope("Actor"):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,  #隐藏层神经元个数
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0,1),  #weight
                bias_initializer = tf.constant_initializer(0.1),  #bias
                name = 'l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs = l1,
                units = n_actions,   #输出个数
                activation= tf.nn.softmax,  #得到行为的可能性
                kernel_initializer=tf.random_normal_initializer(0,1),   #weight
                bias_initializer= tf.constant_initializer(0.1),   #bias
                name = 'acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0,self.a])   #log 动作概率，以ｅ为低的自然对数
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)   #log 概率 * TD方向，调整下次的选择

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  #minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        #s,a用于产生gradient ascent的方向
        #td来自critic,用于告诉actor这方向对不对
        s = s[np.newaxis, :]   #转成一行
        feed_dict = {self.s:s, self.a:a, self.td_error:td}
        _,exp_v = self.sess.run([self.train_op, self.exp_v],feed_dict)
        return exp_v
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s:s})  #得到所有行为的概率情况
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()) #返回一个int类型行为，此处的p参数表示的是前者选择的概率可能性情况

class Critic(object):
    def __init__(self, sess, n_features, lr = 0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32,[1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1,1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,   #隐藏层神经元的个数
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0,1),   #weight
                bias_initializer = tf.constant_initializer(0.1),   #bias
                name = 'l1'
            )

            self.v = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation=None,
                kernel_initializer= tf.random_normal_initializer(0, 1),  #weights
                bias_initializer= tf.constant_initializer(0.1),  #bias
                name = 'V'
            )
        
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    #TD_error = (r+gamma*V_next) - V_eval
        
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :],s_[np.newaxis, :]

        v_ = self.sess.run(self.v,{self.s:s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                        {self.s: s,self.v_:v_, self.r : r})
        return td_error

if __name__ == "__main__":
    sess = tf.Session()
    actor = Actor(sess, n_features = N_F, n_actions = N_A, lr=LR_A)
    critic = Critic(sess, n_features = N_F, lr=LR_C)
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter('logs/',sess.graph)
    
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()
            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            if done:
                r = -20
            track_r.append(r)

            td_error = critic.learn(s, r, s_)  #gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)   #true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():   #globals函数返回一个全局变量的字典，包括所有的导入的变量
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print("episode:",i_episode,"  reward:",int(running_reward))
                break
