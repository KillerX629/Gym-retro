import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


"""Tamaño de la entrada:
    224  320  3
    height, width, channels
    
número de acciones posibles:
    8
    [izq,der]
"""
#tf.image.rgb_to_grayscale(imagen) --> convierte una imagen a escala de grises

class model:
    def __init__(self):
        self.model = keras.FunctionalModel()
        
        #buscar como pasar de imagen de color a blanco y negro!
        self.inputImage = keras.Input(shape=(224, 320, 1))
        #las dimensiones de la entrada son entonces:
        # 224x320 y en vez de 3 canales (RGB), pasamos a blanco y negro
        
        self.inputPastInputs = keras.Input(shape=(10, 8))
        #esta entrada va a tener un vector con los ultimos 10 botones que apretó el agente, en orden cronológico
        #el 8 es por las acciones posibles
        
        #capas de convolución
        self.Conv = keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu') (self.inputImage)
        
        