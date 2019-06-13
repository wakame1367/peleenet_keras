from keras.layers import Input, AveragePooling2D, Dense, Flatten
from keras.models import Model
from layers import stem_block, dense_block, transition_block


def pelee_net(input_shapes=(3, 224, 224), growth_rate=32, num_init_features=32,
              block_configs=None, bottleneck_widths=None, num_classes=10):
    if bottleneck_widths is None:
        bottleneck_widths = [1, 2, 4, 4]
    if block_configs is None:
        block_configs = [3, 4, 8, 6]

    channel, height, width = input_shapes
    x = Input(shape=(channel, height, width))

    _stem_block = stem_block(x, num_init_features)
    total_filter = num_init_features

    for idx, (block_config, bottleneck_width) in enumerate(zip(block_configs,
                                                               bottleneck_widths)):
        out = dense_block(_stem_block, num_layers=block_config,
                          bn_size=bottleneck_width, growth_rate=growth_rate)
        total_filter += growth_rate * block_config

        if idx == len(block_config) - 1:
            with_pooling = False
        else:
            with_pooling = True

        out = transition_block(out, total_filter, with_polling=with_pooling)
        _stem_block = out

    out = AveragePooling2D(pool_size=(7, 7))
    out = Flatten()(out)
    # TODO
    # add Dropout?
    outputs = Dense(num_classes, activation="softmax")(out)
    model = Model(inputs=x, outputs=outputs)

    return model
