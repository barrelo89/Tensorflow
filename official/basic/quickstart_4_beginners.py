import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

#load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Input Image Data Shape: {}".format(x_train.shape))

#build a multi-layer perceptron
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28, 28)), tf.keras.layers.Dense(128, activation = 'relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)])

#probability method 1
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#define the loss function that we will use in training + we set 'from_logits' to True
#by using 'SparseCategoricalCrossentropy', we do not need to do 'one-hot' encoding for label
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

#model compilation: define optimizer, loss function, and performance metrics
model.compile(optimizer = 'adam', loss = loss_fn, metrics = ['accuracy'])

#feed model with training dataset and train the model for 5 epochs
model.fit(x_train, y_train, epochs = 2)

prediction = model(x_test[:1]).numpy()
print("Prediction: {}, Ground Truth: {}".format(np.argmax(prediction), y_test[0]))

#probability method 1
probability = tf.nn.softmax(prediction).numpy()





























#end
