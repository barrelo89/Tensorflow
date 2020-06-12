import os
import tensorflow as tf
from tensorflow import keras

def create_model():

    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation = 'relu', input_shape = (28*28, )))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10))

    #compile the model
    model.compile(optimizer = 'adam', loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

    return model

#load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#create the checkpoint: save weights only
checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

#build the model
model = create_model()
model.summary()

'''
save weights only
if not os.path.exists(checkpoint_dir):

    #create a callback that saves the model's weights
    #Note that by enabling 'save_weights_only', we need to reconstruct the exactly same graph to reuse the saved weights
    #By changing 'period', we can set how frequently we save the trained model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1, period = 5)

    #To save trained model in every 'period' epoch, need to write 'save_weights'
    model.save_weights(checkpoint_path.format(epoch = 0))

    #train the model
    model.fit(train_images, train_labels, epochs = 50, validation_data = (test_images, test_labels), callbacks = [cp_callback])

else:
    loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
    print("Untrained Model Accuracy: {}".format(accuracy))

    model.load_weights(checkpoint_path)

    _, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
    print("Restored Model Accuracy: {}".format(accuracy))
'''

'''
save the entire model
'''
#create the savedmodel directory: save the entire model
model_save_path = 'saved_model/model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

if os.listdir(model_save_path):
    model = tf.keras.models.load_model(model_save_path)
    model.summary()

    loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
    print("Accuracy: {}".format(accuracy))

else:
    model.fit(train_images, train_labels, epochs = 5)
    model.save(model_save_path)

















































#end
