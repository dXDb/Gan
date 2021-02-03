# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 04:33:06 2021

@author: CVPR
"""


# Labraries
from typeguard import typechecked

import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt


# Maxout Activation
class Maxout(tf.keras.layers.Layer):
    @typechecked
    def __init__(self, num_units: int, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tf.shape(inputs)[i]

        num_channels = shape[self.axis]
        if not isinstance(num_channels, tf.Tensor) and num_channels % self.num_units:
            raise ValueError(
                "number of features({}) is not "
                "a multiple of num_units({})".format(num_channels, self.num_units)
            )

        if self.axis < 0:
            axis = self.axis + len(shape)
        else:
            axis = self.axis
        assert axis >= 0, "Find invalid axis: {}".format(self.axis)

        expand_shape = shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        outputs = tf.math.reduce_max(
            tf.reshape(inputs, expand_shape), axis, keepdims=False
        )
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[self.axis] = self.num_units
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {"num_units": self.num_units, "axis": self.axis}
        base_config = super().get_config()
        return {**base_config, **config}


# Model
class myModel(tf.keras.Model):
    def __init__(self, channels_axis):
        super(myModel, self).__init__()
        self.G = self.createG()
        self.D = self.createD(channels_axis)

    def createG(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(784)
        ])


    def createD(self, channels_axis):
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            # Adam 1e-5
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        '''
            tf.keras.layers.Dense(512),
            Maxout(512),
            tf.keras.layers.Dense(256),
            Maxout(256),
            tf.keras.layers.Dense(1)
            '''


# Compute loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def compute_Gloss(fake, images, noise):
    G = cross_entropy(np.ones([fake.shape[0], 1]).astype('float32'), fake)
    return G

def compute_Dloss(real, fake, images, noise):
    D = cross_entropy(np.ones([real.shape[0], 1]).astype('float32'), real)
    G = cross_entropy(np.zeros([fake.shape[0], 1]).astype('float32'), fake)
    return D + G


# Train step
@tf.function
def train_step(model, images, noise, optimizerD, optimizerG):
    for cnt in range(K):
        with tf.GradientTape() as Dtape, tf.GradientTape() as Gtape:
            real = model.D(images)
            fake = model.D(model.G(noise))
            lossD = compute_Dloss(real, fake, images, noise)
            lossG = compute_Gloss(fake, images, noise)

        gradientsD = Dtape.gradient(lossD, model.D.trainable_variables)
        optimizerD.apply_gradients(zip(gradientsD, model.D.trainable_variables))

    gradientsG = Gtape.gradient(lossG, model.G.trainable_variables)
    optimizerG.apply_gradients(zip(gradientsG, model.G.trainable_variables))

    return lossD, lossG

# GradientTape 안에서 동작하는 것들 알아두기, 안에서 꼭 해야하는 것들 알아두기
# gardient 알아두기, apply_gradients알아두기



# Main
IAMGECHANNEL = 1
LEARNINGRATE = 1e-5
EPOCH        = 1000
BATCHSIZE    = 256
K            = 1

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(BATCHSIZE)
test_ds  = tf.data.Dataset.from_tensor_slices(( x_test,  y_test)).batch(BATCHSIZE)

optimizerG = tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE)
optimizerD = tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE)

model = myModel(IAMGECHANNEL)

for epoch in range(1, EPOCH + 1):
    for images, _ in train_ds:
        noise = tf.random.normal(shape=(BATCHSIZE, 100))
        lossD, lossG = train_step(model, images, noise, optimizerD, optimizerG)

    plt.imshow(model.G(noise).numpy()[0].reshape(28, 28) * 255.0, cmap='gray')
    plt.show()
    print("Epoch :", epoch, "D :", lossD.numpy(), "G :", lossG.numpy())