import os
import tempfile

import matplotlib.pyplot as plt
import tensorflow as tf
import visualkeras
from tensorflow import keras
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import Xception, InceptionResNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, GlobalAvgPool2D
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add
from tensorflow.keras.models import Model, model_from_json


def attach_final_layers(model, classes):
    """
        This function takes in a model and the number of classes and returns the model with the final layers attached.
        The final layers are a Global Average Pooling layer, a Dropout layer and a Dense layer with the number of classes
        as the number of neurons.

        Parameters:
        model (keras.Model): The model to which the final layers are to be attached.
        classes (int): The number of classes in the dataset.

        Returns:
        keras.Model: The model with the final layers attached.
    """
    model = GlobalAvgPool2D()(model)
    model = Dropout(rate=0.4)(model)

    activation = 'softmax' if classes > 2 else 'sigmoid'
    classes = 1 if classes == 2 else classes

    output_layer = Dense(classes, activation=activation,
                         bias_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),
                         activity_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))(model)

    return output_layer


def add_regularization(model, l1, l2):
    """
        This function adds regularization to a model.

        Parameters
        ----------
        model : keras.models.Model
            The model to add regularization to.
        l1 : float
            The l1 regularization coefficient.
        l2 : float
            The l2 regularization coefficient.

        Returns
        -------
        keras.models.Model
            The model with regularization.
    """
    regularizer = regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def get_EfficientNetB0(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB0(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB1(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB1(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB2(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB2(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB3(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    """
    This function returns a compiled EfficientNetB3 model.

    Parameters
    ----------
    weights : str
        The path to the weights file to be loaded or "imagenet" to load pre-trained network.
        If None, the model will be initialized with random weights.
    input_shape : tuple
        The shape of the input layer.
    classes : int
        The number of classes in the output layer.
    regularize : bool
        Whether to add regularization to the model.
    l1 : float
        The L1 regularization coefficient.
    l2 : float
        The L2 regularization coefficient.

    Returns
    -------
    model : keras.Model
        A compiled EfficientNetB3 model.
    """
    model = EfficientNetB3(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB4(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB4(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB5(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB5(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB6(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB6(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_EfficientNetB7(weights=None, input_shape=(96, 96, 3), classes=8, regularize=True, l1=0.00001, l2=0.00001):
    model = EfficientNetB7(include_top=False,
                           weights=weights,
                           input_shape=input_shape,
                           classes=8)

    model.trainable = True

    if regularize:
        model = add_regularization(model, l1, l2)

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_InceptionResNetV2(weights=None, input_shape=(96, 96, 3), classes=8):
    model = InceptionResNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        classes=8)

    model.trainable = True

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def get_Xception(weights=None, input_shape=(96, 96, 3), classes=8):
    model = Xception(include_top=False,
                     weights=weights,
                     input_shape=input_shape,
                     classes=8)

    model.trainable = True

    input_layer = Input(shape=input_shape)
    x = preprocess_input(input_layer)

    model = model(x)

    output_layer = attach_final_layers(model, classes)

    return Model(inputs=input_layer, outputs=output_layer)


def cbr(input_net, filters, kernel_size, strides):
    """
        Convolutional Block with ReLU activation and Dropout
        Args:
            input_net: input tensor
            filters: number of filters
            kernel_size: kernel size
            strides: strides
        Returns:
            net: output tensor
    """
    net = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_uniform',
                 kernel_regularizer=regularizers.l2(0.0001),
                 strides=strides, padding='same')(input_net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(rate=0.2)(net)
    return net


def skip_blk(input_net, filters, kernel_size=3, strides=1):
    """
        This function is used to create a skip block.
        It takes in the input_net, filters, kernel_size, strides as parameters.
        It returns the skip block.
    """
    net = cbr(input_net=input_net, filters=filters, kernel_size=kernel_size, strides=strides)
    net = cbr(input_net=net, filters=filters, kernel_size=kernel_size, strides=strides)
    net = cbr(input_net=net, filters=filters, kernel_size=kernel_size, strides=strides)
    skip = cbr(input_net=input_net, filters=filters, kernel_size=kernel_size, strides=strides)
    net = Add()([skip, net])
    net = cbr(input_net=net, filters=filters, kernel_size=3, strides=strides * 2)
    return net


def customcnn(input_shape=(96, 96, 3), classes=8, filters=None):
    """
    This function creates a custom CNN model.
    The model is a simple CNN with skip connections.
    """

    if filters is None:
        filters = [32, 64, 128]
    input_layer = Input(shape=input_shape)
    net = input_layer

    for f in filters:
        net = skip_blk(net, f)

    # reduce channels
    f = int(filters[-1])
    net = skip_blk(net, f)

    net = Conv2D(filters=f, kernel_size=(3, 3), kernel_initializer='he_uniform',
                 kernel_regularizer=regularizers.l2(0.0001), strides=1, padding='same')(net)

    net = Dropout(rate=0.2)(net)

    net = GlobalAvgPool2D()(net)
    net = Dropout(0.2)(net)
    net = Dense(256, activation='relu',
                kernel_initializer=initializers.GlorotUniform(),
                bias_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.00001))(net)
    net = Dropout(0.2)(net)
    net = Dense(128, activation='relu',
                kernel_initializer=initializers.GlorotUniform(),
                bias_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.00001))(net)
    net = Dropout(0.2)(net)
    output_layer = Dense(classes, activation='softmax',
                         kernel_initializer=initializers.GlorotUniform(),
                         bias_regularizer=regularizers.l2(0.0001),
                         activity_regularizer=regularizers.l2(0.00001))(net)

    return Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    model = get_EfficientNetB3()

    print(model.summary())

    plt.imshow(visualkeras.layered_view(model, legend=True, scale_xy=15))
    plt.show()
    tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=False, to_file='model.png')
