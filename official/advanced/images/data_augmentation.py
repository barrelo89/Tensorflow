import numpy as np
import tensorflow as tf
import matplotlib as mlp
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers

def visualize(original, augmented):

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.axis('off')
    plt.title('Augmented')

    plt.show()

def augment(image, label):

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_crop_or_pad(image, 34, 34)
    image = tf.image.random_crop(image, size = [28, 28, 1]) # Random crop back to 28x28
    image = tf.image.random_brightness(image, max_delta = 0.5) # Random brightness

    return image,label

def convert(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def build_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28, 28, 1)))
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10))

    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

    return model

mlp.rcParams['figure.figsize'] = (12, 5)

AUTOTUNE = tf.data.experimental.AUTOTUNE

image_path = tf.keras.utils.get_file('cat.jpg',  "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
image = plt.imread(image_path)

'''
plt.imshow(image)
plt.axis('off')
plt.tight_layout()
plt.show()
'''

#flipping the image
flipped = tf.image.flip_left_right(image)
#visualize(image, flipped)

#grayscale the image
grayscaled = tf.image.rgb_to_grayscale(image)
#visualize(image, tf.squeeze(grayscaled))

#saturate the image: saturation ->  채도인듯
saturated = tf.image.adjust_saturation(image, 3)
#visualize(image, saturated)

#change the image brightness
bright = tf.image.adjust_brightness(image, 0.4)
#visualize(image, bright)

#rotate the image
rotated = tf.image.rot90(image, k = 1)
#visualize(image, rotated)

#center crop the image
cropped = tf.image.central_crop(image, central_fraction = 0.5)
#visualize(image, cropped)

#train a model with augmented dataset
dataset, info =  tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples= info.splits['train'].num_examples

BATCH_SIZE = 64
NUM_EXAMPLES = 2048

augmented_train_batches = (train_dataset.take(NUM_EXAMPLES).cache().shuffle(num_train_examples // 4).map(augment, num_parallel_calls = AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))
non_augmented_train_batches = (train_dataset.take(NUM_EXAMPLES).cache().shuffle(num_train_examples // 4).map(convert, num_parallel_calls = AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))

validation_batches = (test_dataset.map(convert, num_parallel_calls = AUTOTUNE).batch(BATCH_SIZE))

#without augmentation
model_without_aug = build_model()
training_without_aug = model_without_aug.fit(non_augmented_train_batches, validation_data = validation_batches, epochs = 50)

#with augmentation
model_with_aug = build_model()
training_with_aug = model_with_aug.fit(augmented_train_batches, epochs = 50, validation_data = validation_batches)

history_without_aug = training_without_aug.history
history_with_aug = training_with_aug.history

plt.plot(history_without_aug['accuracy'], label = 'TRAINING ACCURACY WITHOUT AUGMENTATION')
plt.plot(history_without_aug['val_accuracy'], label = 'VALIDATION ACCURACY WITHOUT AUGMENTATION')
plt.plot(history_with_aug['accuracy'], label = 'TRAINING ACCURACY WITH AUGMENTATION')
plt.plot(history_with_aug['val_accuracy'], label = 'VALIDATION ACCURACY WITH AUGMENTATION')
plt.title('ACCURACY')
plt.ylim([0.75, 1])
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

plt.plot(history_without_aug['loss'], label = 'TRAINING LOSS WITHOUT AUGMENTATION')
plt.plot(history_without_aug['val_loss'], label = 'VALIDATION LOSS WITHOUT AUGMENTATION')
plt.plot(history_with_aug['loss'], label = 'TRAINING LOSS WITH AUGMENTATION')
plt.plot(history_with_aug['val_loss'], label = 'VALIDATION LOSS WITH AUGMENTATION')
plt.title('Loss')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()



























#end
