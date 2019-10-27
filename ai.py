import tensorflow as tf
import numpy as np
import pdb
from random import random, randrange, sample


# tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float64')
EPISLON = 0.01
EPISLON_DECAY = 0.95
GAMMA = 0.95
LR = 0.005
INPUT_SIZE = 4
OUTPUT_SIZE = 2
BATCH_SIZE = 32
MEMORY_MAX = 1000000


class Model(tf.keras.Model):
    """Subclassing du réseau de neuronnes"""

    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer()
        self.dense_layer1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear', name='result')

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return self.output_layer(x)

    def pick_action(self, state, eps):
        global EPISLON
        if random() < eps:  # On ne suit pas la policy et donc on choisit une action aléatoire
            print('ALéatoire')
            action = randrange(OUTPUT_SIZE)  # On choisit une action aléatoire
            EPISLON = EPISLON * EPISLON_DECAY
            return action
        else:
            action = self.predict(state)  # On passe le state dans le réseau de neuronne
            action = np.argmax(action)  # On choisit l'index de l'action avec la valeur la plus haut
            return action


class Memory:
    """Implémenatation de la mémoire des actions de l'agent"""

    def __init__(self, batch_size, memory_max):
        self.storage = []  # variable de stockage
        self.batch_size = batch_size
        self.memory_max = memory_max

    def add(self, state, next_state, action, reward, done):
        if self.memory_max == len(self.storage):  # On supprimer le premier élément en mémoire si on atteint la capacité max
            del self.storage[0]
        self.storage.append([state, next_state, action, reward, done])  # On enrengistre les actions dans la mémoire

    def get_sample(self):
        batch = list(zip(*sample(self.storage, self.batch_size)))  # On regroupe les éléments en liste de state, liste de next_state, liste d'action et liste de reward

        for i1 in range(1):  # Boucle de suppression d'une dimension inutile
            for i2 in range(len(batch[i1]) - 1):
                np.squeeze(batch[i1][i2], axis=0)  # Suppresion de la dimension, données sous la forme

        batch[0] = tf.convert_to_tensor(batch[0], dtype=tf.float64)  # on transforme en tensor
        batch[1] = tf.convert_to_tensor(batch[1], dtype=tf.float64)  # voir au dessus
        batch[2] = tf.expand_dims(tf.convert_to_tensor(batch[2], dtype=tf.float64), 1)  # on transforme en tensor et ajoute une dimension la compatibilté des shapes pour les calculs
        batch[3] = tf.expand_dims(tf.convert_to_tensor(batch[3], dtype=tf.float64), 1)  # voir au dessus
        batch[4] = tf.expand_dims(tf.convert_to_tensor(batch[4], dtype=tf.float64), 1)

        return batch

    def is_enough_data(self):
        return len(self.storage) >= BATCH_SIZE  # renvoie True si on a assez de données pour l'entrainement

    def last_reward(self):
        return self.storage[-1][3]  # renvoie le dernier reward


class DQN:
    """Implémentation du Deep Q Learning"""

    def __init__(self):
        self.model = Model()  # Notre modèle (réseau de neuronne)
        self.optimizer = tf.keras.optimizers.Adam(lr=LR)  # l'optimizer pour la back propagation
        self.memory = Memory(BATCH_SIZE, MEMORY_MAX)  # Mémoire des actions de l'agent
        self.reward = 0
        self.state = np.array([[0.0, 0.0, 0.0, 0.0]])  # Etat actuelle
        self.next_state = np.array([[0.0, 0.0, 0.0, 0.0]])  # Etat suivant
        self.metrics_loss = tf.metrics.MeanSquaredError()

    @tf.function
    def train(self, batch_states, batch_next_states, batch_actions, batch_reward, batch_done):

        next_action_max = tf.reduce_max(self.model(batch_next_states)) * (1 - batch_done) # Q(s', a', 0)
        q_targets = tf.add(batch_reward, tf.scalar_mul(GAMMA, next_action_max))  # Calcul de la target, r + GAMMA * Q(s', a', 0)*
        # pdb.set_trace()

        with tf.GradientTape() as tape:  # On prépare le calcul du gradient
            predictions = tf.reduce_max(self.model(batch_states))  # Q(s, a, 0)
            loss = tf.keras.losses.MSE(q_targets, predictions)  # Calcul de l'erreur

        grads = tape.gradient(loss, self.model.trainable_variables)  # Calcul du gradient
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # On applique le gradient à notre modèle

        self.metrics_loss(q_targets, predictions)

    def select_action(self, signals, reward, done):
        self.state = self.next_state  # On récupère l'état actuelle
        self.next_state = np.array([signals])  # On récupère le nouvelle état
        action = self.model.pick_action(self.next_state, EPISLON)  # On choisit une action

        self.memory.add(self.state, self.next_state, action, reward, done)  # On ajoute une transition à la mémoire

        if self.memory.is_enough_data():  # On entraine notre modèle si il y a assez de données
            batch_states, batch_next_states, batch_actions, batch_reward, batch_done = self.memory.get_sample()  # On récupère notre lot de données
            self.train(batch_states, batch_next_states, batch_actions, batch_reward, batch_done)
            print('------------------------------------------')
            print("la moyenne des erreurs est de %s" % self.metrics_loss.result())  # On affiche la moyenne des erreurs non cumulées
            print("le reward est de %s" % self.memory.last_reward())  # On affiche le dernier rewarc
            print('------------------------------------------')

        return action
