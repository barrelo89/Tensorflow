import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

'''
SINCE I AM WORKING ON PYTHON 3.5, I CANNOT FULLY UTILIZE TENSORFLOW_DATASETS DUE TO PYTHON VERSION ISSUE (ESPECIALLY 'CATS VS DOGS' DATASET)
TO USE A 'CATS VS DOGS' DATASET, WE NEED TO MODIFY THE SOURCE CODE OF TENSORFLOW_DATASETS:
/usr/local/lib/python3.5/dist-packages/tensorflow_datasets/core/download/extractor.py, line 199, in iter_zip
    if member.is_dir():  # Filter directories
AttributeError: 'ZipInfo' object has no attribute 'is_dir'

https://gist.github.com/invisiblefunnel/673520cd4d2bc5a5bfb79c158c1885c8
Need to change it as follows:
member.is_dir() -> member.filename[-1] == '/'
'''

IMG_SIZE = 160

def format_example(image, label):

    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label

#Load the 'cats_vs_dogs' dataset
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info = True, as_supervised = True)

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
plt.show()

train, validation, test = raw_train.map(format_example), raw_validation.map(format_example), raw_test.map(format_example)

BATCH_SIZE, SHUFFLE_BUFFER_SIZE = 32, 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    print(image_batch.shape)

#TRANSFER LEARNING
base_model = tf.keras.applications.MobileNetV2(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top = False, weights = 'imagenet')

base_learning_rate = 0.0001
initial_epoch = 10
validation_steps = 20

#MODEL EXTRACT FEATURES FROM INPUT IMG
feature_sample = base_model(image_batch)
print(feature_sample.shape)

#1. DISABLE TRAINABLE JUST TO USE IT AS A FEATURE EXTRACTOR
base_model.trainable = False
base_model.summary()


#BUILD A MODEL
model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(1)])
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate), loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
model.summary()

#TRAINING
initial_loss, initial_accuracy = model.evaluate(validation_batches, steps = validation_steps)
print("Initial Loss: {}, Initial Accuracy: {}".format(initial_loss, initial_accuracy))

training = model.fit(train_batches, epochs = initial_epoch, validation_data = validation_batches)
history = training.history

accuracy, loss = history['accuracy'], history['loss']
val_accuracy, val_loss = history['val_accuracy'], history['val_loss']
'''
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()
'''

#FINE TUNING: we will retrain some top layers of the transferred neural network
base_model.trainable = True
print("Number of Layers in the Base Model: {}".format(len(base_model.layers)))

tuning_start_at = 100

for layer in base_model.layers[:tuning_start_at]:
    layer.trainable = False

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate / 10), metrics = ['accuracy'])
model.summary()

fine_tune_epoch = 10
total_epochs = initial_epoch + fine_tune_epoch
fine_training = model.fit(train_batches, epochs = total_epochs, initial_epoch = training.epoch[-1], validation_data = validation_batches)

accuracy += fine_training.history['accuracy']
loss += fine_training.history['loss']

val_accuracy += fine_training.history['val_accuracy']
val_loss += fine_training.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epoch-1, initial_epoch-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epoch-1, initial_epoch-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()























#end
