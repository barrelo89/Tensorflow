import os
import PIL
import time
import glob
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display
from tensorflow.keras import layers

def make_generator_model():

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', use_bias = False, activation = 'tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
#discriminator loss
def discriminator_loss(real_output, fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss

#Generator Loss
def generator_loss(fake_output):

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training = False)

plt.imshow(generated_image[0, :, :, 0], cmap = 'gray')
plt.axis('off')
plt.show()

discriminator = make_discriminator_model()
decision = discriminator(generated_image)

#Loss and optimizers
generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_opt, discriminator_optimizer = discriminator_opt, generator = generator, discriminator = discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 10

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training = True)

        real_prediction = discriminator(images, training = True)
        fake_prediction = discriminator(generated_images, training = True)

        gen_loss = generator_loss(fake_prediction)
        disc_loss = discriminator_loss(real_prediction, fake_prediction)

    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):

    for epoch in range(epochs):

        start_time = time.time()

        for batch in dataset:

            train_step(batch)

        display.clear_output(wait = True)
        generate_and_save_image(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print("Time of Epoch {}: {} sec".format(epoch + 1, time.time() - start_time))

    display.clear_output(wait = True)
    generate_and_save_image(generator, epochs, seed)

def generate_and_save_image(model, epoch, test_input):

    fig = plt.figure(figsize = (4,4))

    prediction = model(test_input, training = False)

    for idx in range(prediction.shape[0]):

        plt.subplot(4, 4, idx + 1)
        plt.imshow(tf.squeeze(prediction[idx]))
        plt.axis('off')

    plt.savefig('dcgan/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

train(train_dataset, EPOCHS)

anim_file = 'dcgan/dcgan.gif'

with imageio.get_writer(anim_file, mode = 'I') as writer:

    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)

    last = -1

    for idx, name in enumerate(filenames):

        image = imageio.imread(name)
        writer.append_data(image)

display.Image(filename = anim_file)
















































#end
