import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10, 10))

for idx in range(25):

    plt.subplot(5, 5, idx+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[idx], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[idx][0]])
plt.show()

model = tf.keras.Sequential()

#Convolution Layer
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = train_images[0].shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

#Dense Layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.summary()

#compile the model: optimizer and loss function
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])


#training
training = model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs = 10)

#visualize the result
history = training.history

plt.plot(history['accuracy'], 'b-', label = 'accuracy')
plt.plot(history['val_accuracy'], 'r-*', label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

loss, accuracy = model.evaluate(test_images, test_labels, verbose = 1)


































#end
