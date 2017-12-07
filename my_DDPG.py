# DDPG: image input --> action, image, action --> q value
# environment: 2_link robot arm

import tensorflow as tf
import numpy as np
import os
import shutil
from arm_env import ArmEnv
from collections import deque
import random
import cv2

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 10000
MAX_EP_STEPS = 200 
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64
VAR_MIN = 0.1
RENDER = True
LOAD = False
MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTIONS = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, 80, 80, 1], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, 80, 80, 1], name='s_')
with tf.name_scope('S1'):
    S1 = tf.placeholder(tf.float32, shape=[None, 80, 80, 4], name='s1')
with tf.name_scope('R1'):
    R1 = tf.placeholder(tf.float32, [None, 1], name='r1')
with tf.name_scope('S1_'):
    S1_ = tf.placeholder(tf.float32, shape=[None, 80, 80, 4], name='s1_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            W_conv1 = self.weight_variable([8, 8, 1, 300])
            b_conv1 = self.bias_variable([300])
            W_conv2 = self.weight_variable([4, 4, 300, 300])
            b_conv2 = self.bias_variable([300])
            W_conv3 = self.weight_variable([3, 3, 300, 300])
            b_conv3 = self.bias_variable([300])
            W_fc1 = self.weight_variable([7500, 512])
            b_fc1 = self.bias_variable([512])
            W_fc2 = self.weight_variable([512, ACTIONS])
            b_fc2 = self.bias_variable([ACTIONS])
            #s = tf.placeholder("float", [None, 80, 80, 4])
            h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            #h_pool2 = max_pool_2x2(h_conv2)
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #h_pool3 = max_pool_2x2(h_conv3)
            #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
            h_conv3_flat = tf.reshape(h_conv3, [-1, 7500])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            with tf.variable_scope('a'):
                actions = tf.matmul(h_fc1, W_fc2) + b_fc2
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        #self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S1, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S1_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R1 + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            W_conv1 = self.weight_variable([8, 8, 4, 32])
            b_conv1 = self.bias_variable([32])
            W_conv2 = self.weight_variable([4, 4, 32, 64])
            b_conv2 = self.bias_variable([64])
            W_conv3 = self.weight_variable([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])
            W_fc1 = self.weight_variable([1600, 512])
            b_fc1 = self.bias_variable([512])
            W_fc2 = self.weight_variable([512, ACTIONS])
            b_fc2 = self.bias_variable([ACTIONS])
            #s = tf.placeholder("float", [None, 80, 80, 4])
            h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            #h_pool2 = max_pool_2x2(h_conv2)
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #h_pool3 = max_pool_2x2(h_conv3)
            #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            with tf.variable_scope('q'):
                q = tf.matmul(h_fc1, W_fc2) + b_fc2
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S1: s, self.a: a, R1: r, S1_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

D = deque()

saver = tf.train.Saver()
path = './'+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 2.  # control exploration

    for ep in range(MAX_EPISODES):
        pos_s = env.reset()
        env.render()
        env.step([0.,0.])
        env.render()
        x_t = env.getBinaryImage()
        #cv2.imshow('show',x_t)
        #cv2.waitKey(100)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
        # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = actor.choose_action(np.reshape(s_t[:,:,0], (80, 80, 1)))
            #a = np.clip(np.random.normal(a, var), *ACTION_BOUND)
            a = np.random.normal(a, var)   # add randomness to action selection for exploration
            pos_s_, r, done = env.step(a)

            env.render()
            x_t1 = env.getBinaryImage()
            #cv2.imshow('show',x_t1)
            #cv2.waitKey(100)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            D.append((s_t,a,r,s_t1))

            if len(D) > MEMORY_CAPACITY:
                D.popleft()
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                minibatch = random.sample(D,BATCH_SIZE)
                b_s = np.array([d[0] for d in minibatch])
                b_a = np.array([d[1] for d in minibatch])
                b_r = np.array([d[2] for d in minibatch]).reshape((len(minibatch),1))
                b_s_= np.array([d[3] for d in minibatch])

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(np.reshape(b_s[:,:,:,0], (len(b_s),80, 80, 1)))

            s_t = s_t1
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
            # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    env.set_fps(30)
    pos_s = env.reset()
    env.render()
    env.step([0.,0.])
    env.render()
    x_t = env.getBinaryImage()
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(np.reshape(s_t[:,:,0], (80, 80, 1)))
        pos_s_, r, done = env.step(a)
        env.render()
        x_t1 = env.getBinaryImage()
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
