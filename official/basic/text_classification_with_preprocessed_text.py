import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tensorflow import keras

tfds.disable_progress_bar()

#load the dataset
(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', split = (tfds.Split.TRAIN, tfds.Split.TEST), as_supervised = True, with_info = True)

#Let's see how the encoder (text to vector) works
encoder = info.features['text'].encoder
print("Vocabulary Size in Encoder: {}".format(encoder.vocab_size))

sample_string = 'Hello Tensorflow'

encoded_string = encoder.encode(sample_string)
print("Original: {} -> Encoded: {}".format(sample_string, encoded_string))

decoded_string = encoder.decode(encoded_string)
print("Encoded: {} -> Decoded: {}".format(encoded_string, decoded_string))

assert sample_string == decoded_string

for word_vector in encoded_string:
    print("Word Vector: {}, Word: {}".format(word_vector, encoder.decode([word_vector])))


for train_text, train_label in train_data.take(1): #train_data.take(1) == train_data[0]

    print("Embedding Vectors: {}".format(train_text))
    print("Original Review: {}".format(encoder.decode(train_text)))
    print("Review Sentiment: {}".format(train_label))

#create the training and test batches
BUFFER_SIZE = 1000

train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32))
test_batches = (test_data.padded_batch(32))

#build the model
model = keras.Sequential()
model.add(keras.layers.Embedding(encoder.vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(1))

model.summary()

#compile the model: set the optimizer and loss function
model.compile(optimizer = 'adam', loss = tf.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])

#train the model
training = model.fit(train_batches, epochs = 10, validation_data = test_batches, validation_steps = 30)

#Let's see how the trained model works
loss, accuracy = model.evaluate(test_batches)

print("Loss: {}, Accuracy: {}".format(loss, accuracy))

#visualze the accuracy and loss over time
history = training.history
print(history.keys())

acc, val_acc, loss, val_loss = history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()









































#end
