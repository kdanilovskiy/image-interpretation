import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Dense, Activation, Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, UpSampling1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D

from tensorflow.keras.layers import Reshape, Flatten, Lambda, Permute, PReLU

from tensorflow.keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

def conv2d_down_block(
        layer,
        filters=1,
        kernel_size=(1, 1),
        activation='relu',
        dropout=None,
        batch_norm=True,
        seed=37,
        max_pooling=None,
        average_pooling=None,
        upsampling=None,
        padding='same',
):
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        activation=None,
        use_bias=True,
        padding=padding,
    )(layer)

    if str(activation).lower() == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)

    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout, seed=seed)(x) if dropout else x
    x = MaxPooling2D(pool_size=max_pooling)(x) if max_pooling else x
    x = AveragePooling2D(pool_size=average_pooling)(x) if average_pooling else x
    x = UpSampling2D(size=upsampling)(x) if upsampling else x
    return x


def conv2d_block(
        layer,
        filters=1,
        kernel_size=1,
        activation='relu',
        dropout=None,
        batch_norm=True,
        seed=37,
):
    x = Conv2D(
        filters,
        kernel_size=(3, 3),
        activation=None,
        use_bias=True,
        padding='same'
    )(layer)

    if str(activation).lower() == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)

    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout, seed=seed)(x) if dropout else x
    return x


def dense_block(
        layer,
        units,
        activation='relu',
        dropout=None,
        batch_norm=False,
        seed=37,
):
    x = Dense(
        units,
        activation=None,
        use_bias=True
    )(layer)

    if str(activation).lower() == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    x = BatchNormalization()(x) if batch_norm else x
    x = Dropout(dropout, seed=seed)(x) if dropout else x
    return x