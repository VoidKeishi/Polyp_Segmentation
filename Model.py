import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_0 = layers.Conv2D(filters, 1, strides, 'same')
        self.conv_1 = layers.Conv2D(filters, kernel_size, strides, 'same')
        self.batch_norm_1 = layers.BatchNormalization()
        self.relu_1 = layers.ReLU()
        self.conv_2 = layers.Conv2D(filters, kernel_size, strides, 'same')
        self.batch_norm_2 = layers.BatchNormalization()
        self.add = layers.Add()
        self.relu_2 = layers.ReLU()

    def call(self, x):
        res = self.conv_0(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.add([x, res])
        x = self.batch_norm_2(x)
        x = self.relu_2(x)
        return x

# Define the U-Net model
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Downsample path
    conv1 = ResidualBlock(64, 3)(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ResidualBlock(128, 3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ResidualBlock(256, 3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ResidualBlock(512, 3)(pool3)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = ResidualBlock(1024, 3)(pool4)
    drop5 = Dropout(0.5)(conv5)

    # Upsample path
    up6 = ResidualBlock(512, 3)(UpSampling2D(size=(2, 2))(drop5))
    conv6 = concatenate([drop4, up6], axis=3)

    up7 = ResidualBlock(256, 3)(UpSampling2D(size=(2, 2))(conv6))
    conv7 = concatenate([conv3, up7], axis=3)

    up8 = ResidualBlock(128, 3)(UpSampling2D(size=(2, 2))(conv7))
    conv8 = concatenate([conv2, up8], axis=3)

    up9 = ResidualBlock(64, 2)(UpSampling2D(size=(2, 2))(conv8))
    conv9 = concatenate([conv1, up9], axis=3)
    conv9 = ResidualBlock(32, 3)(conv9)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the model architecture
def load_model():
    input_shape = (128, 128, 3)  # Adjust input shape based on your image size and channels
    model = unet_model(input_shape)
    model.load_weights('./Weights/unet_model_weight_gg.h5')
    return model