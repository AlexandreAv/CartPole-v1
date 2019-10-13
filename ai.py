import tensorflow as tf
import numpy as np
import pdb
from random import random, randrange, sample


# tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float64')
EPISLON = 0.01
GAMMA = 0.95
LR = 0.0005
INPUT_SIZE = 4
OUTPUT_SIZE = 2
BATCH_SIZE = 32
MEMORY_MAX = 1000000


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer()
        self.dense_layer1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear', name='result')
        self.optimize = tf.compat.v1.train.AdamOptimizer(1e-4)

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return self.output_layer(x)

    def pick_action(self, state, eps):
        if random() < eps:
            print('ALéatoire')
            action = randrange(OUTPUT_SIZE)
            return action
        else:
            action = self.predict(state)
            action = np.argmax(action)
            return action


class Memory:
    def __init__(self, batch_size, memory_max):
        self.storage = []
        self.batch_size = batch_size
        self.memory_max = memory_max

    def add(self, state, next_state, action, reward):
        if self.memory_max == len(self.storage):
            del self.storage[0]
        self.storage.append([state, next_state, action, reward])

    def get_sample(self):
        batch = list(zip(*sample(self.storage, self.batch_size)))

        for i1 in range(1):
            for i2 in range(len(batch[i1]) - 1):
                np.squeeze(batch[i1][i2], axis=0)

        batch[0] = tf.convert_to_tensor(batch[0], dtype=tf.float64)
        batch[1] = tf.convert_to_tensor(batch[1], dtype=tf.float64)
        batch[2] = tf.expand_dims(tf.convert_to_tensor(batch[2], dtype=tf.float64), 1)
        batch[3] = tf.expand_dims(tf.convert_to_tensor(batch[3], dtype=tf.float64), 1)

        return batch

    def is_enough_data(self):
        return len(self.storage) >= BATCH_SIZE

    def last_reward(self):
        return self.storage[-1][3]


class DQN:
    def __init__(self):
        self.model = Model()
        self.optimizer = tf.keras.optimizers.Adam(lr=LR)
        self.memory = Memory(BATCH_SIZE, MEMORY_MAX)
        self.reward = 0
        self.state = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.next_state = np.array([[0.0, 0.0, 0.0, 0.0]])

    def train(self):
        batch_states, batch_next_states, batch_actions, batch_reward = self.memory.get_sample()

        next_action_max = tf.reduce_max(self.model(batch_next_states))
        q_targets = tf.add(batch_reward, tf.scalar_mul(GAMMA, next_action_max))
        # pdb.set_trace()

        with tf.GradientTape() as tape:
            predictions = tf.reduce_max(self.model(batch_states))
            loss = tf.keras.losses.MSE(q_targets, predictions)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        print('------------------------------------------')
        print("l'erreur est de %s" % tf.reduce_mean(loss))
        print("le reward est de %s" % self.memory.last_reward())
        print('------------------------------------------')

    def select_action(self, signals, reward):
        self.state = self.next_state
        self.next_state = np.array([signals])
        action = self.model.pick_action(self.next_state, EPISLON)

        self.memory.add(self.state, self.next_state, action, reward)

        if self.memory.is_enough_data():
            self.train()

        return action
