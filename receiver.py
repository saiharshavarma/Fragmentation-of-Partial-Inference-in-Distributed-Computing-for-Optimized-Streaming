from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)


def create_submodel(input_size):
    submodel = tf.keras.Sequential()
    submodel.add(tf.keras.layers.Dense(1000000))
    submodel.add(tf.keras.layers.Activation('relu'))
    return submodel


@app.route('/', methods=['POST'])
def data_transference():
    submodel_inputs = request.data

    submodel = create_submodel(submodel_inputs.shape[1])
    processed_data = submodel.predict(submodel_inputs)

    return processed_data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
