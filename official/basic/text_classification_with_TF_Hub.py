import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("TF Version: {}".format(tf.__version__))
print("Eager Mode Enabled?: {}".format(tf.executing_eagerly()))
print("Hub Version: {}".format(hub.__version__))
print("GPU IS ", "AVAILABLE" if tf.config.experimental.list_physical_devices('GPU') else "Not Available.")

train_data, validation_data, test_data = tfds.load(name = "imdb_reviews", split = ('train[:60%]', 'train[60%:]', 'test'), as_supervised = True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

#Using Transfer Learning: use a pre-trained text embedding model
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" #Output Dimension: 20
hub_layer = hub.KerasLayer(embedding, input_shape = [], dtype = tf.string, trainable = True)

#build the model
model =tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

#show the summary of the model
model.summary()

#compile the model: seetting the optimizer, loss function, and performance metrics
model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])

#train the model
history = model.fit(train_data.shuffle(10000).batch(512), epochs = 20, validation_data = validation_data.batch(512), verbose = 1)

results = model.evaluate(test_data.batch(512), verbose = 2)

for metric_name, result in zip(model.metrics_names, results):

    print("{}: {}".format(metric_name, result))







































#end
