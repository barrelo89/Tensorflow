import numpy as np
import tensorflow as tf
import PIL.Image as Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from tensorflow.keras import layers

#https://www.tensorflow.org/guide/keras/custom_callback
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs = None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['accuracy'])
        self.model.reset_metrics()

image_shape = (224, 224)

'''
#download the classifier
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
classifier = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape = image_shape + (3,))])

#download the corresponding label and a sample data
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(image_shape)
grace_hopper = np.array(grace_hopper) / 255.0

#Prediction
prediction = classifier.predict(grace_hopper[np.newaxis, ...])
predicted_class = np.argmax(prediction[0])

plt.imshow(grace_hopper)
plt.axis('off')
plt.title("Prediction: {}".format(imagenet_labels[predicted_class].title()))
plt.show()
'''

#Transfer Learning
data_root = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
image_data = image_generator.flow_from_directory(data_root, target_size = image_shape) #default batch_size = 32

(image_batch, label_batch) = image_data.next()
'''
prediction = classifier.predict(image_batch)

predicted_class = imagenet_labels[np.argmax(prediction, axis = 1)]

#VISUALIZE THE RESULT
plt.figure(figsize = (10, 9))
plt.subplots_adjust(hspace = 0.5)

for idx in range(30):

    plt.subplot(6, 5, idx + 1)
    plt.imshow(image_batch[idx])
    plt.title(predicted_class[idx])
    plt.axis('off')
    plt.suptitle('ImageNet Predictions')
plt.show()
'''

#Use a model without the top classification layer
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape = (224, 224, 3))

'''
FREEZE THE VARIABLES IN THE FEATURE EXTRACTOR LAYER, SO THAT THE TRAINING ONLY MODIFIES THE NEW CLASSIFIER LAYER
'''
feature_extractor_layer.trainable = False

#add a calssification head
model = tf.keras.Sequential([feature_extractor_layer, layers.Dense(image_data.num_classes)])
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)
batch_stats_callback = CollectBatchStats()

training = model.fit_generator(image_data, epochs = 2, steps_per_epoch = steps_per_epoch, callbacks = [batch_stats_callback])

plt.plot(batch_stats_callback.batch_losses)
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.show()

plt.plot(batch_stats_callback.batch_acc)
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.show()

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis = -1)
predicted_label_batch = class_names[predicted_id]

print(label_batch)
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
    plt.suptitle("Model predictions (green: correct, red: incorrect)")

plt.show()


















#end
