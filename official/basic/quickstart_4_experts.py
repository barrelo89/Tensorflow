import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class EM(Model):

    def __init__(self):
        super(EM, self).__init__()
        self.conv1 = Conv2D(32, 3, activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation = 'relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

mnist = tf.keras.datasets.mnist

#load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#since Conv2D requires 3D dataset, we add a channel dimension
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = EM()

#define loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')


#tf.function: compile a function into a callable Tensorflow graph [https://www.tensorflow.org/api_docs/python/tf/function]
@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:

        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):

    predictions = model(images, training = False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

#Training
epochs = 5

for epo in range(1, epochs+1):

    train_loss.reset_states()
    train_accuracy.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for images, labels in test_ds:
        test_step(images, labels)

    print("Epoch: {}, Training Loss: {}, Training Accuracy: {}, Test Loss: {}, Test Accuracy: {}".format(epo, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))












































































#end
