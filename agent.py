import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import random
from collections import deque


