import tensorflow as tf
import keras as K

"""
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Convolution2D, Flatten
from keras.callbacks import TensorBoard
import numpy as np
import random
from collections import deque
from rl.agents import DQNAgent
#from rl.memory import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
"""

"""Tamaño de la entrada:
    224  320  3
    height, width, channels
"""


def build_model(height,width,channels,actions):
    model=K.models.Sequential()
    model.add(K.Models.InputLayer(input_shape=(height,width,channels)))
    model.add(K.models.Convolution2D(32,(8,8),strides=(4,4),input_shape=(4, height,width,channels),activation='relu'))
    model.add(K.models.Convolution2D(64,(4,4),strides=(2,2),activation='relu'))
    model.add(K.models.Convolution2D(64,(3,3),strides=(1,1),activation='relu'))
    model.add(K.models.Flatten())
    model.add(K.models.GRU(512,activation='relu',return_sequences=True))# error en la entrada, forma incompatible con capa anterior
    model.add(K.models.GRU(256,activation='relu'))
    model.add(K.models.GRU(128,activation='relu'))
    model.add(K.models.GRU(actions,activation='linear'))
    return model



def build_agent(model,actions):
    policy = K.distribute.LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=1000000)
    memory = SequentialMemory(limit=10000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True,dueling_type='avg', nb_actions=actions)
    return dqn

