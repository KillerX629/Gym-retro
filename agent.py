import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Convolution2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import random
from collections import deque
from rl.agents import DQNAgent
from rl.memory import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

def build_model(height,width,channels,actions):
    model=Sequential()
    model.add(Convolution2D(32,(8,8),strides=(4,4),input_shape=(4, height,width,channels),activation='relu'))
    model.add(Convolution2D(64,(4,4),strides=(2,2),activation='relu'))
    model.add(Convolution2D(64,(3,3),strides=(1,1),activation='relu'))
    model.add(Flatten())
    model.add(GRU(512,activation='relu',return_sequences=True))
    model.add(GRU(256,activation='relu'))
    model.add(GRU(128,activation='relu'))
    model.add(GRU(actions,activation='linear'))
    return model
"""este modelo no esta probado, los modelos de Youtube están con capas convolucionales y densas"""


def build_agent(model,actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=1000000)
    memory = SequentialMemory(limit=10000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True,dueling_type='avg', nb_actions=actions)
    return dqn

