import keras.backend as K
from keras.layers import (Conv2D, BatchNormalization, Concatenate, ReLU,
                          MaxPooling2D, AveragePooling2D)


def dense_layer(x, growth_rate, bottleneck_width=4, name="2way-dense-layer"):
    """ 3-way inputs
    1. right: conv_block_1
    2. middle: raw_output from previous_layer
    3. left: conv_block_2

    :return:
    """
    num_input_features = K.int_shape(x)[-1]
    _middle = x
    _growth_rate = int(growth_rate / 2)
    _inter_channel = int(_growth_rate * bottleneck_width / 4) * 4

    if _inter_channel > num_input_features / 2:
        _inter_channel = int(num_input_features / 8) * 4
        print('adjust inter_channel to ', _inter_channel)

    #
    conv_block_left1 = conv_block(x, _inter_channel, kernel_size=1, strides=1,
                                  padding="valid", )
    conv_block_left2 = conv_block(conv_block_left1, _growth_rate,
                                  kernel_size=3, strides=1,
                                  padding="same")

    #
    conv_block_right1 = conv_block(x, _inter_channel, kernel_size=1, strides=1,
                                   padding="valid", )
    conv_block_right2 = conv_block(conv_block_right1, _growth_rate,
                                   kernel_size=3, strides=1,
                                   padding="same")
    conv_block_right3 = conv_block(conv_block_right2, _growth_rate,
                                   kernel_size=3, strides=1,
                                   padding="same")
    _out = Concatenate()([_middle, conv_block_right3, conv_block_left2])
    return _out


def dense_block(x, num_layers, growth_rate, bn_size):
    inputs = x
    for layer_idx in range(num_layers):
        inputs = dense_layer(inputs, growth_rate, bn_size)
    return inputs


def conv_block(inputs, out_channels, kernel_size, strides, padding,
               activation=True):
    x = inputs
    conv = Conv2D(out_channels, kernel_size=kernel_size,
                  strides=strides, padding=padding, use_bias=False)
    x = conv(x)
    # reference
    # https://github.com/qxcv/caffe2keras/blob/eed4e4b17743888426f1f8439279bf558c8d62e4/caffe2keras/convert.py#L359
    x = BatchNormalization(epsilon=0.001, momentum=0.999)(x)

    if activation:
        x = ReLU()(x)
    return x


def stem_block(inputs, num_init_features):
    """

    :return:
    """
    x = inputs
    #
    stem1 = conv_block(x, num_init_features, kernel_size=3, strides=2,
                       padding="same")
    stem2 = conv_block(stem1, int(num_init_features / 2), kernel_size=1,
                       strides=1,
                       padding="valid")
    stem2 = conv_block(stem2, num_init_features, kernel_size=3, strides=2,
                       padding="same")
    #
    max_pooling_2d = MaxPooling2D(pool_size=(2, 2), strides=2)
    stem1 = max_pooling_2d(stem1)
    concat = Concatenate()([stem1, stem2])
    stem3 = conv_block(concat, num_init_features, kernel_size=1, strides=1,
                       padding="valid")

    return stem3


def transition_block(inputs, num_filter, with_polling=True):
    x = inputs
    conv = conv_block(x, num_filter, kernel_size=1, strides=1,
                      padding="valid")
    if with_polling:
        average_polling_2d = AveragePooling2D(pool_size=(2, 2), strides=2)
        # max_pooling_2d = MaxPooling2D(pool_size=(2, 2), strides=2)
        pooling = average_polling_2d(conv)
        out = pooling
    else:
        out = conv
    return out
