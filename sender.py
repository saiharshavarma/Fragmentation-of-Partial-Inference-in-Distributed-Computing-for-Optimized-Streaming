import tensorflow as tf
import numpy as np
import requests


def create_model(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_shape=(input_size,)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(1000000))
    model.add(tf.keras.layers.Activation('relu'))
    return model


input_size = 100
output_size = 10

model = create_model(input_size, output_size)

num_samples = 1000
X_data = np.random.normal(size=(num_samples, input_size))

submodel_inputs = X_data
for layer in model.layers[:-1]:
    submodel = tf.keras.models.Model(model.input, layer.output)
    submodel_inputs = submodel.predict(submodel_inputs)

response = requests.post('http://device2_address:port', data=submodel_inputs)

process_processed_data(response.content)
