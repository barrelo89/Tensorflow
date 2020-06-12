import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#path_to_zip: save the path to the directory containing 'cats_and_dogs.zip' file
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin = _URL, extract = True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_path = os.path.join(PATH, 'train')
train_cat_path = os.path.join(train_path, 'cats')
train_dog_path = os.path.join(train_path, 'dogs')

validation_path = os.path.join(PATH, 'validation')
validation_cat_path = os.path.join(validation_path, 'cats')
validation_dog_path = os.path.join(validation_path, 'dogs')

num_dogs_tr, num_cats_tr, num_dogs_val, num_cats_val = len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path))
total_tr = num_dogs_tr + num_cats_tr
total_val = num_dogs_val + num_cats_val

print("Num of Train Dogs Imgs: {}, Num of Train Cats Imgs: {}, Num of Validation Dogs Imgs: {}, Num of Validation Cats Imgs: {}".format(len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path)), len(os.listdir(train_dog_path))))

batch_size = 128
epochs = 15
IMG_HEIGHT, IMG_WIDTH = 150, 150

#DATA PREPARATION
train_image_generator = ImageDataGenerator(rescale = 1. / 255)
validation_image_generator = ImageDataGenerator(rescale = 1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size = batch_size, directory = train_path, shuffle = True, target_size = (IMG_HEIGHT, IMG_WIDTH), class_mode = 'binary')
validation_data_gen = validation_image_generator.flow_from_directory(batch_size = batch_size, directory = validation_path, target_size = (IMG_HEIGHT, IMG_WIDTH), class_mode = 'binary')

#visualize training images
sample_training_images, sample_training_labels = next(train_data_gen)

plotImages(sample_training_images)


#BUILD THE MODEL
model = Sequential()
model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1))

#COMPILE THE MODEL: OPTIMIZER AND LOSS FUNCTION
model.compile(optimizer = 'adam', loss = tf.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
model.summary()

#TRAIN THE MODEL
training = model.fit_generator(train_data_gen, steps_per_epoch = total_tr // batch_size, epochs = epochs, validation_data = validation_data_gen, validation_steps = total_val // batch_size)

#VISUALIZE TRAINING RESULTS
history = training.history

epoch_range = range(epochs)

plt.subplot(1, 2, 1)
plt.plot(epoch_range, history['accuracy'], 'b--', label = 'TRAINING ACCURACY')
plt.plot(epoch_range, history['val_accuracy'], 'r-*', label = 'VALIDATION ACCURACY')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, history['loss'], 'b--', label='Training Loss')
plt.plot(epoch_range, history['val_loss'], 'r-*', label='Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()


'''
'flow_from_directory' method return:
tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels
'''
#DATA AUGMENTATION
#1. horizontal flip

'''
image_gen = ImageDataGenerator(rescale = 1. / 255. horizontal_flip = True )
horizontal_flip_train_data_gen = image_gen.flow_from_directory(batch_size = batch_size, directory = train_path, shuffle = True, target_size = (IMG_HEIGHT, IMG_WIDTH))
augmented_images = [horizontal_flip_train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#2. randomly rotate the image
image_gen = ImageDataGenerator(rescale = 1. / 255, rotation_range = 45)
rotation_train_data_gen = image_gen.flow_from_directory(batch_size = batch_size, directory = train_path, shuffle = True, target_size = (IMG_HEIGHT, IMG_WIDTH))

augmented_images = [rotation_train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#3. zoom augmentation
image_gen = ImageDataGenerator(rescale = 1. / 255, zoom_range = 0.5)
zoom_train_data_gen = image_gen.flow_from_directory(batch_size = batch_size, shuffle = True, directory = train_path, target_size = (IMG_HEIGHT, IMG_WIDTH))

augmented_images = [zoom_train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
'''

#APPLY DATA AUGMENTATION N DROPOUT
train_data_gen = ImageDataGenerator(rescale = 1. / 255, horizontal_flip = True, rotation_range = 45, width_shift_range = .15, height_shift_range = .15, zoom_range = 0.5)
train_data = train_data_gen.flow_from_directory(batch_size = batch_size, shuffle = True, directory = train_path, target_size = (IMG_HEIGHT, IMG_WIDTH), class_mode = 'binary')

validation_data_gen = ImageDataGenerator(rescale = 1. / 255)
validation_data = validation_data_gen.flow_from_directory(batch_size = batch_size, directory = validation_path, target_size = (IMG_HEIGHT, IMG_WIDTH), class_mode = 'binary')

new_model = Sequential()
new_model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
new_model.add(MaxPooling2D())
new_model.add(Dropout(0.3))
new_model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
new_model.add(MaxPooling2D())
new_model.add(Dropout(0.3))
new_model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
new_model.add(MaxPooling2D())
new_model.add(Dropout(0.3))

new_model.add(Flatten())
new_model.add(Dense(512, activation = 'relu'))
new_model.add(Dense(1))

new_model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])
new_model.summary()

training = new_model.fit_generator(train_data, steps_per_epoch = total_tr // batch_size, validation_data = validation_data, validation_steps = total_val // batch_size, epochs = epochs)

history = training.history

epoch_range = range(epochs)

plt.subplot(1, 2, 1)
plt.plot(epoch_range, history['accuracy'], 'b--', label = 'TRAINING ACCURACY')
plt.plot(epoch_range, history['val_accuracy'], 'r-*', label = 'VALIDATION ACCURACY')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, history['loss'], 'b--', label='Training Loss')
plt.plot(epoch_range, history['val_loss'], 'r-*', label='Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()
















#end
