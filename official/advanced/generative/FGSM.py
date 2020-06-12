import tensorflow as tf
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)


def preprocess(image):

    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]

    return image

def get_image_labels(probability):
    return tf.keras.applications.mobilenet_v2.decode_predictions(probability, top = 1)[0][0]

'''
tf.GradientTape().watch(input_tensor):
ensures that the input_tensor is being traced by this tape
'''

def create_adversarial_pattern(model, input_image, input_label):

    with tf.GradientTape() as tape:
        tape.watch(input_image)

        prediction = model(input_image)
        loss = tf.keras.losses.CategoricalCrossentropy()(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_gradient = tf.sign(gradient)

    return signed_gradient

def display_image(model, image, description):

    prediction_probability = model.predict(image)
    _, image_class, class_confidence = get_image_labels(prediction_probability)

    plt.imshow(image[0]*0.5 + 0.5)
    plt.title('{} \n classificed as {} in {} confidence'.format(description, image_class, class_confidence))
    plt.axis('off')
    plt.show()

pretrained_model = tf.keras.applications.MobileNetV2(include_top = True, weights = 'imagenet')
pretrained_model.trainable = False

image_path = tf.keras.utils.get_file('input.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image)

image = preprocess(image)
image_probability = pretrained_model.predict(image)

_, image_class, class_confidence = get_image_labels(image_probability)

plt.imshow(image[0]*0.5 + 0.5)
plt.title('{}: {} confidence'.format(image_class, class_confidence))
plt.axis('off')
plt.show()

loss_function = tf.keras.losses.CategoricalCrossentropy()

labrado_index = 208
label = tf.one_hot(labrado_index, image_probability.shape[-1])
label = tf.reshape(label, (1, -1))

perturbation = create_adversarial_pattern(pretrained_model, image, label)
plt.imshow(perturbation[0]*0.5 + 0.5)
plt.axis('off')
plt.show()

epsilone_list = [0.0, 0.01, 0.1, 0.15]

descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilone_list]

for idx, eps in enumerate(epsilone_list):

    adv_x = image + eps*perturbation
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_image(pretrained_model, adv_x, descriptions[idx])






































#end
