import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix

def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask

@tf.function
def load_image_train(datapoint):

    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:

        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):

    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def display(display_list):

    #plt.figure(figsize = (15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for idx, img in enumerate(display_list):

        plt.subplot(1, len(display_list), idx + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
        plt.axis('off')
        plt.title(title[idx])
    plt.tight_layout()
    plt.show()

def unet_model(output_channels):

    input = tf.keras.layers.Input(shape = [128, 128, 3])
    x = input

    #Downsampling through the model: feature extraction
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Upsampling and establishing the skip connections
    for  up, skip in zip(up_stack, skips):

        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    #Last Layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides = 2, padding = 'same') #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs = input, outputs = x)

def create_mask(pred_mask):

    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis] # (None, 128, 128) -> (None, 128, 128, 1)

    return pred_mask[0]

def show_predictions(dataset = None, num = 1):

    if dataset:

        for image, mask in dataset.take(num):

            pred_mask = model.predict(image)
            display([image[0], mask[0], pred_mask])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info = True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

test_dataset = test.batch(BATCH_SIZE)

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
#display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs = base_model.input, outputs = layers)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

tf.keras.utils.plot_model(model, to_file = 'model.png', show_shapes = True)

#show_predictions()
class DisplayCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait = True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE

training = model.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = test_dataset, validation_steps = VALIDATION_STEPS, callbacks = [DisplayCallback()])

loss = training.history['loss']
val_loss = training.history['val_loss']

epochs = range(EPOCHS)

plt.plot(epochs, loss, 'r--', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b-*', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
plt.show()









































#end
