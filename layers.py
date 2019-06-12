from keras.layers import Conv2D, BatchNormalization, Concatenate, ReLU


def dense_block(x, growth_rate, bottleneck_width=4, name="2way-dense-layer"):
    """ 3-way inputs
    1. right: conv_block_1
    2. middle: raw_output from previous_layer
    3. left: conv_block_2

    :return:
    """
    _middle = x
    _growth_rate = int(growth_rate / 2)
    _inter_channel = int(_growth_rate * bottleneck_width / 4) * 4

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
