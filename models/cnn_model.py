from pyexpat import model

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


class CustomCNN(tf.keras.Model): # creation de la classe CustomCNN
    def __init__(self,num_classes,**kwargs):
        super(CustomCNN,self).__init__(**kwargs) #  heritage de CustomCNN a la classe tf.keras.Model
        self.num_classes = num_classes
        # premiere couche de convolution
        self.conv1=layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=1,
            padding='same',
            activation='relu',
        )
        self.bn1=layers.BatchNormalization()
        self.pool1=layers.MaxPool2D( #On garde l'essentielle
            pool_size=2,
            strides=2,
        )
        # Deuxieme couche de convolution 
        self.conv2=layers.Conv2D(
            filters=256,
            kernel_size=(3,3),
            strides=1,
            padding='same',
            activation='relu',
        )
        self.bn2=layers.BatchNormalization()
        self.pool2=layers.MaxPool2D(
            pool_size=2,
            strides=2,
        )
        # Troisieme couche de convolution 
        self.conv3=layers.Conv2D(
            filters=512,
            kernel_size=(3,3),
            strides=1,
            padding='same',
            activation='relu',
        )
        self.bn3=layers.BatchNormalization()
        self.pool3=layers.MaxPool2D(
            pool_size=2,
            strides=2,
        )
        # couche d'applatissement/flatten
        self.flatten=layers.Flatten()

        self.d1=layers.Dense(64,activation='relu')
        self.d2=layers.Dense(32,activation='relu')
        self.d3=layers.Dense(16,activation='relu')
        self.dp=layers.Dropout(rate=0.5)
        self.d6=layers.Dense(num_classes,activation='softmax')
    
    def call(self,inputs): # Appel des couches du modele lors de l'entrainement
        x=self.conv1(inputs)
        x=self.bn1(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.pool2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.pool3(x)

        x=self.flatten(x)

        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.dp(x)
        return self.d6(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)