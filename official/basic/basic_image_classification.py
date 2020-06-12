import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_labels)

print("Train Imageset Shape: {}, Train Label Shape: {}, Test Imageset Shape: {}, Test Label Shape: {}".format(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape))

#Since labels are in integrer format, make a list containing the corresponding label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images, test_images = train_images / 255.0, test_images / 255.0

#visualize a subset of training dataset
num_fig = 25

plt.figure(figsize = (10, 10))
for idx in range(num_fig):
    plt.subplot(5, 5, idx + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[idx])
    plt.xlabel(class_names[train_labels[idx]])
plt.tight_layout()
plt.show()

#build the simple multi-layer perceptron
num_epochs = 1

model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(128, activation = 'relu'), keras.layers.Dense(10)])
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = num_epochs)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 2)

#produce the probability
probability = keras.Sequential([model, keras.layers.Softmax()])
predictions = probability(test_images)

























































#end
