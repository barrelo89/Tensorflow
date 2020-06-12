import IPython
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras

'''
FOR MORE DETAIL ABOUT TUNER, PLEASE VISIT:
https://keras-team.github.io/keras-tuner/
'''

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

def model_builder(hp):

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape = (28, 28)))

    #Tune the # of units in the first Dense
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
    model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
    model.add(keras.layers.Dense(10))

    #Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate), loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

    return model

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

tuner = kt.Hyperband(model_builder, objective = 'val_accuracy', max_epochs = 10, factor = 3, directory = 'hyper', project_name = 'intro_to_kt')
tuner.search(train_images, train_labels, validation_data = (test_images, test_labels), callbacks = [ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print("The Optimal Number of Units: {}, The Optimal Learning Rate: {}".format(best_hps.get('units'), best_hps.get('learning_rate')))

#build the model with the best hyperparameter
model = tuner.hypermodel.build(best_hps)
model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))













































#end
