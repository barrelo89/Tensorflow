import os
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display

IMG_WIDTH = 256
IMG_HEIGHT = 256

OUTPUT_CHANNELS = 3

def load(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    width = tf.shape(image)[1]

    real_image = image[:, :(width//2), :]
    input_image = image[:, (width//2):, :]

    real_image, input_image = tf.cast(real_image, tf.float32), tf.cast(input_image, tf.float32)

    return real_image, input_image

def resize(image, height, width):
    return tf.image.resize(image, [height, width], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def random_crop(real_image, input_image):

    stacked_image = tf.stack([real_image, input_image], axis = 0)
    cropped_image = tf.image.random_crop(stacked_image, size = [2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1] #cropped_real, cropped_input

def normalize(image):
    return (image - 127.5) / 127.5

@tf.function
def random_jitter(real_image, input_image):
    real_image = resize(real_image, 286, 286)
    input_image = resize(input_image, 286, 286)

    real_image, input_image = random_crop(real_image, input_image)

    if tf.random.normal([]) > 0.5:
        real_image = tf.image.flip_left_right(real_image)
        input_image = tf.image.flip_left_right(input_image)

    return real_image, input_image

def load_image_train(image_path):

    real_image, input_image = load(image_path)
    real_image, input_image = random_jitter(real_image, input_image)
    real_image = normalize(real_image)
    input_image = normalize(input_image)

    return real_image, input_image

def load_image_test(image_path):

    real_image, input_image = load(image_path)

    real_image = resize(real_image, IMG_HEIGHT, IMG_WIDTH)
    real_image = normalize(real_image)

    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize(input_image)

    return real_image, input_image

def downsample(num_filters, filter_size, apply_batchnorm = True):

    initializer = tf.random_normal_initializer(0., 0.02)

    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(num_filters, filter_size, strides = (2, 2), padding = 'same', kernel_initializer = initializer, use_bias = False))

    if apply_batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())
    layer.add(tf.keras.layers.LeakyReLU())

    return layer

def upsample(num_filters, filter_size, apply_dropout = False):

    initializer = tf.random_normal_initializer(0., 0.02)

    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2DTranspose(num_filters, filter_size, strides = (2, 2), padding = 'same', kernel_initializer = initializer, use_bias = False))
    layer.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        layer.add(tf.keras.layers.Dropout(0.5))

    layer.add(tf.keras.layers.ReLU())

    return layer

def Generator():

    inputs = tf.keras.layers.Input(shape = [256, 256, 3])

    down_stack = [
    downsample(64, 4, apply_batchnorm = False), #128 X 128
    downsample(128, 4), #64 X 64 X 128
    downsample(256, 4), #32 X 32 X 256
    downsample(512, 4), #16 X 16 X 512
    downsample(512, 4), #8 X 8 X 512
    downsample(512, 4), #4 X 4 X 512
    downsample(512, 4), #2 X 2 X 512
    downsample(512, 4) #1 X 1 512
    ]

    up_stack = [
    upsample(512, 4, apply_dropout = True), #2 X 2 X 512
    upsample(512, 4, apply_dropout = True), #4 X 4 X 512
    upsample(512, 4, apply_dropout = True), #8 X 8 X 512
    upsample(512, 4, apply_dropout = True), #16 X 16 X 512
    upsample(256, 4), #32 X 32 X 256
    upsample(128, 4), #64 X 64 X 128
    upsample(128, 4), #128 X 128 X 128
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides = (2, 2), use_bias = False, padding = 'same', kernel_initializer = initializer, activation ='tanh')

    x = inputs
    skip_list = []

    for layer in down_stack:
        x = layer(x)
        skip_list.append(x)

    skip_list = reversed(skip_list[:-1])

    for skip_input, up in zip(skip_list, up_stack):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip_input])

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

def generator_loss(disc_generated_output, gen_output, target, LAMBDA):

    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

    total_gen_loss = gen_loss + LAMBDA*l1_loss

    return total_gen_loss, gen_loss, l1_loss

def Discriminator():

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape = [256, 256, 3], name = 'input_img')
    tar = tf.keras.layers.Input(shape = [256, 256, 3], name = 'target_img')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides = 1, kernel_initializer = initializer, use_bias = False)(zero_pad)
    batchnorm = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm)

    zero_pad_1 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    conv_1 = tf.keras.layers.Conv2D(1, 4, strides = 1, kernel_initializer = initializer)(zero_pad_1)

    return tf.keras.Model(inputs = [inp, tar], outputs = conv_1)

def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss

def generate_images(model, test_input, tar):

    prediction = model(test_input, training = True)

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for idx in range(3):

        plt.subplot(1, 3, idx + 1)
        plt.title(title[idx])
        plt.imshow(display_list[idx]*0.5 + 0.5)
        plt.axis('off')
    plt.savefig(str(time.time())+ '.png')
    plt.close()

URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin = URL, extract = True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades')

BUFFER_SIZE = 400
BATCH_SIZE = 1

real, input = load(os.path.join(PATH, 'train/100.jpg'))

train_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'test/*.jpg'))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = Generator()
#tf.keras.utils.plot_model(gen, show_shapes = True, dpi = 64)

discriminator = Discriminator()

#wieght value on the L1 loss between the generated image and the target image
LAMBDA = 100

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#for real, input in test_dataset.take(1):
#    generate_images(generator, input, real)
EPOCHS = 150

log_dir = 'logs/'
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:

        gen_output = generator(input_image, training = True)

        disc_real = discriminator([input_image, target], training = True)
        disc_gen = discriminator([input_image, gen_output], training = True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen, gen_output, target, LAMBDA)
        disc_loss = discriminator_loss(disc_real, disc_gen)

    generator_gradient = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discrminator_gradient = dis_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discrminator_gradient, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):

    for epoch in range(epochs):

        start = time.time()
        display.clear_output(wait = True)

        for example_real, example_input in test_ds.take(1):
            generate_images(generator, example_input, example_real)
        print("EPOCH: {}".format(epoch))

        for idx, (real_image, input_image) in train_ds.enumerate():
            if (idx+1) % 100 == 0:
                print('STEP {} COMPLETED'.format(idx+1))
            train_step(input_image, real_image, epoch)
        print()

        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    checkpoint.save(file_prefix = checkpoint_prefix)

fit(train_dataset, EPOCHS, test_dataset)


























#end
