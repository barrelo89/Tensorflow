import time
import functools
import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as display

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

def tensor_to_image(tensor):

    tensor = tensor*255
    tensor = np.array(tensor, dtype = np.uint8)
    print(tensor.shape)

    return PIL.Image.fromarray(tensor[0])

def load_img(img_path):

    max_dim = 512
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    print(img.shape)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape*scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, ...]

    return img

def imshow(image, title = None):

    if len(image.shape) > 3:
        image = tf.squeeze(image, axis = 0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.tight_layout()

def vgg_layers(layer_names):

    vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model


content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_img = load_img(content_path)
style_img = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_img, "Content Image")

plt.subplot(1, 2, 2)
imshow(style_img, "Style Image")
plt.show()

#multiply 255 since we scaled the input image in 'load_img' function
'''
'tf.keras.applications.vgg19.preprocess_input' returns:
Preprocessed numpy.array or a tf.Tensor with type float32.
The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
'''
x = tf.keras.applications.vgg19.preprocess_input(content_img*255)
x = tf.image.resize(x, (244, 244))

vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')

for layer in vgg.layers:
    print(layer.name)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
content_extractor = vgg_layers(content_layers)

def gram_matrix(input_tensor):

    result = tf.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result / num_locations

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):

        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layer = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):

        inputs = inputs*255
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = outputs[:self.num_style_layer], outputs[self.num_style_layer:]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name:value for content_name, value in zip(content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(content_img)

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())

style_target = extractor(style_img)['style']
content_target = extractor(content_img)['content']

image = tf.Variable(content_img)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)

opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 = 0.99, epsilon = 1e-1)

style_weight = 1e-2
content_weight = 1e4

def style_content_loss(outputs):

    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name])**2) for name in style_outputs.keys()])
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name])**2) for name in content_outputs.keys()])

    style_loss *= style_weight
    content_loss *= content_weight / num_content_layers

    return style_loss + content_loss

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    print(tf.shape(image))


epochs = 10
steps_per_epoch = 100

start_time = time.time()

for epoch in range(epochs):

    for step in range(steps_per_epoch):

        train_step(image)
        print(".", end = '')
        print("Train Epoch: {}, Step: {}/{}".format(epoch+1, step+1, steps_per_epoch))

end_time = time.time()
print("Total Time: {} (s)".format(end_time - start_time))

result = tensor_to_image(image)
plt.imshow(result)
plt.show()

























#end
