import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
    image = tf.image.random_crop(image, size = [IMG_HEIGHT, IMG_WIDTH, 3])
    return image

def normalize(image):
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)

    return image

def preprocess_image_train(image, label):

    image = random_jitter(image)
    image = normalize(image)

    return image

def preprocess_image_test(image, label):
    image = normalize(image)
    return image

def discriminator_loss(real, generated):

    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.zeros_like(generated), generated)

    return 0.5*(real_loss + generated_loss)

def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(generated), generated)

def cal_cycle_loss(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))
    return LAMBDA*loss

def identity_loss(real, generated):
    return LAMBDA*0.5*tf.reduce_mean(tf.abs(real - generated))

def generate_images(model, test_input):

    prediction = model(test_input)
    img_list = [test_input[0], prediction[0]]
    title_list = ['Input', 'Generated']

    for idx, img in enumerate(img_list):

        plt.subplot(1, 2, idx + 1)
        plt.imshow(img*0.5 + 0.5)
        plt.title(title_list[idx])
        plt.axis('off')
    plt.savefig(str(time.time()) + '.png')
    plt.close()


dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info = True, as_supervised = True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000
BATCH_SIZE = 1
OUTPUT_CHANNELS = 3
LAMBDA = 10

train_horses = train_horses.map(preprocess_image_train, num_parallel_calls = AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls = AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(preprocess_image_test, num_parallel_calls = AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls = AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

y_generator = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type = 'instancenorm')
x_generator = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type = 'instancenorm')

y_discriminator = pix2pix.discriminator(norm_type = 'instancenorm', target = False)
x_discriminator = pix2pix.discriminator(norm_type = 'instancenorm', target = False)

generated_y = y_generator(sample_horse)
generated_x = x_generator(sample_zebra)

prediction_y = y_discriminator(generated_y)
prediction_x = x_discriminator(generated_x)

y_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
x_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

y_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
x_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

EPOCHS = 40

@tf.function
def train_step(x, y):

    with tf.GradientTape(persistent = True) as tape:

        generated_y = y_generator(x, training = True)
        cycled_x = x_generator(generated_y, training = True)

        generated_x = x_generator(y, training = True)
        cycled_y = y_generator(generated_x, training = True)

        identity_y = y_generator(y, training = True)
        identity_x = x_generator(x, training = True)

        disc_real_y = y_discriminator(y, training = True)
        disc_real_x = x_discriminator(x, training = True)

        disc_generated_y = y_discriminator(generated_y, training = True)
        disc_generated_x = x_discriminator(generated_x, training = True)

        #discriminator loss
        x_discriminator_loss = discriminator_loss(disc_real_x, disc_generated_x)
        y_discriminator_loss = discriminator_loss(disc_real_y, disc_generated_y)

        #generator loss
        y_generator_loss = generator_loss(disc_generated_y)
        x_generator_loss = generator_loss(disc_generated_x)

        #cycle loss
        cycle_loss = cal_cycle_loss(x, cycled_x) + cal_cycle_loss(y, cycled_y)

        #identity loss
        y_generator_identity_loss = identity_loss(y, identity_y)
        x_generator_identity_loss = identity_loss(x, identity_x)

        #total generator loss
        total_y_generator_loss = y_generator_loss + cycle_loss + y_generator_identity_loss
        total_x_generator_loss = x_generator_loss + cycle_loss + x_generator_identity_loss

    y_generator_gradients = tape.gradient(total_y_generator_loss, y_generator.trainable_variables)
    x_generator_gradients = tape.gradient(total_x_generator_loss, x_generator.trainable_variables)

    y_discriminator_gradients = tape.gradient(y_discriminator_loss, y_discriminator.trainable_variables)
    x_discriminator_gradients = tape.gradient(x_discriminator_loss, x_discriminator.trainable_variables)

    y_generator_optimizer.apply_gradients(zip(y_generator_gradients, y_generator.trainable_variables))
    x_generator_optimizer.apply_gradients(zip(x_generator_gradients, x_generator.trainable_variables))

    y_discriminator_optimizer.apply_gradients(zip(y_discriminator_gradients, y_discriminator.trainable_variables))
    x_discriminator_optimizer.apply_gradients(zip(x_discriminator_gradients, x_discriminator.trainable_variables))

for epoch in range(EPOCHS):

    start = time.time()

    for idx, (img_x, img_y) in enumerate(tf.data.Dataset.zip((train_horses, train_zebras))):

        if (idx + 1) % 10 == 0:
            print("TRAINING STEP {} COMPLETED".format(idx + 1))

        train_step(img_x, img_y)

    clear_output(wait = True)

    generate_images(y_generator, sample_horse)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))




























#end
