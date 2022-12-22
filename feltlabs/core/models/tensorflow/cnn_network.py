import re
from typing import Tuple

import numpy as np
import tensorflow as tf


def _get_cnn(layer, cnn):
    """Construct CNN layers based on architecture definition."""
    while cnn:
        layer_name, params = cnn.pop(0).split("-", 1)
        if layer_name == "C":
            filters, k_size, stride, padding = params.split("-")
            layer = tf.keras.layers.Conv2D(
                int(filters), int(k_size), int(stride), padding, activation=tf.nn.relu
            )(layer)
        elif layer_name == "CB":
            filters, k_size, stride, padding = params.split("-")
            layer = tf.keras.layers.Conv2D(
                int(filters), int(k_size), int(stride), padding, use_bias=False
            )(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)
        elif layer_name == "M":
            k_size, stride = params.split("-")
            layer = tf.keras.layers.MaxPool2D(int(k_size), int(stride))(layer)
        elif layer_name == "R":
            residual = _get_cnn(layer, params[1:-1].split(","))
            layer = tf.keras.layers.Add()([residual, layer])
        elif layer_name == "F":
            layer = tf.keras.layers.Flatten()(layer)
        elif layer_name == "H":
            h_size = params
            layer = tf.keras.layers.Dense(int(h_size), activation=tf.nn.relu)(layer)
        elif layer_name == "D":
            dropout_rate = params
            layer = tf.keras.layers.Dropout(float(dropout_rate))(layer)
    return layer


def get_cnn_network(
    architecture: str, input_shape: Tuple[int, int, int], classes: int
) -> tf.keras.Model:
    """Create CNN network based on specified architecture,

    Args:
        architecture: comma-separated list of layers, longer description below
        input_shape: tuple containing (height, width, channels)
        classes: number of final outputs/classes

    Returns:
        keras model which can be used for training and prediction

    Layers description:
        Architecture is comma-separated list of the following layers:
        - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
          activation and specified number of filters, kernel size, stride and padding.
          Example: `C-10-3-1-same`
        - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
          In detail, start with a convolutional layer without bias and activation,
          then add batch normalization layer, and finally ReLU activation. Example: `CB-10-3-1-same`
        - `M-kernel_size-stride`: Add max pooling with specified size and stride. Example: `M-3-2`
        - `R-[layers]`: Add a residual connection. The `layers` contain a specification
          of at least one convolutional layer (but not a recursive residual connection `R`).
          The input to the specified layers is then added to their output
          (after the ReLU nonlinearity of the last one). Example: `R-[C-16-3-1-same,C-16-3-1-same]`
        - `F`: Flatten inputs. Must appear exactly once in the architecture.
        - `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size. Example: `H-100`
        - `D-dropout_rate`: Apply dropout with the given dropout rate. Example: `D-0.5`
        We assume that provided architecture is valid, else it will fail.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Produce the results in variable `hidden`.
    # Split by commas which aren't in brackets []
    cnn = re.split(r"(?<!\[[^[\]]),(?![^[\]]*\])", architecture)
    hidden = _get_cnn(inputs, cnn)
    # Add the final output layer
    outputs = tf.keras.layers.Dense(classes, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # TODO: Test if necessary - should init weights
    model(np.expand_dims(np.random.random(input_shape), axis=0))
    return model
