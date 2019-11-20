import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Dense, Activation, Input, Conv2D
from tensorflow.keras.layers import Reshape, Flatten, Lambda, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from .layers import *
from .metrices import *


def model_classification(
        n_classes,
        shape=None,
        filters=32,
        kernel_size=32,
        last_kernel_size=32,
        activation='relu',
        last_activation='sigmoid',
        dropout=.1,
        batch_norm=True,
        lr=.001,
        decay=0.1,
):
    metrics = [ssim, 'accuracy', precision, recall, fbeta_score]
    if not isinstance(filters, list):
        filters = [filters]

    if not isinstance(kernel_size, list):
        kernel_size = [kernel_size]

    if not isinstance(activation, list):
        activation = [activation]

    _input = Input(shape=(shape, shape))

    x = _input
    x = Lambda(lambda x: K.expand_dims(x))(x)

    for _filters, _kernel_size, _activation in zip(filters, kernel_size, activation):
        x = conv2d_block(
            x,
            _filters,
            [_kernel_size] * 2,
            activation=_activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    x = Conv2D(
        n_classes,
        kernel_size=[last_kernel_size] * 2,
        use_bias=True,
        activation=None,
        padding='same'
    )(x)

    x = Activation(last_activation)(x)

    model = Model(_input, x, name="conv_segm")

    optimizer = Adam(lr=lr, decay=decay, amsgrad=False)

    if n_classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model