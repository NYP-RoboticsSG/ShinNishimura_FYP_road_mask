import tensorflow as tf

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())

    return result

def create_gan_model():
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))

    down_stack = [
        downsample(24,  4, apply_batchnorm=False),
        downsample(32,  4),
        downsample(48,  4),
        downsample(52,  4),
        downsample(64,  4),
        downsample(96,  4),
        downsample(128, 4),
    ]
    up_stack = [
        upsample(96, 4, apply_dropout=True),
        upsample(64, 4, apply_dropout=True),
        upsample(52, 4, apply_dropout=True),
        upsample(48, 4),
        upsample(32, 4),
        upsample(24, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh'
    )

    x = inputs
    x = tf.image.resize(x, (128, 128))
    x = tf.image.rgb_to_hsv(x)[..., 1:4]


    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def create_main_model():
    def model(input_data):
        output_data = tf.keras.layers.Dropout(0.25)(input_data)
        # (None, 96, 96, 1)

        output_data = tf.keras.layers.Conv2D(
            16, kernel_size=9, strides=(3, 3), padding='same', activation='relu')(output_data)
        # (None, 32, 32, 8)

        output_data = tf.keras.layers.Conv2D(
            32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output_data)
        # (None, 16, 16, 16)

        output_data = tf.keras.layers.Conv2D(
            64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(output_data)
        # (None, 8, 8, 32)

        output_data = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(2, 2), padding='same')(output_data)
        # (None, 4, 4, 32)

        output_data = tf.keras.layers.Flatten()(output_data)
        # (None, 512)

        output_data = tf.keras.layers.Dense(
            64, activation='relu')(output_data)
        # (None, 32)

        output_data = tf.keras.layers.Dense(
            2, activation='tanh')(output_data)
        # (None, 2)
        return output_data
    input_layer = tf.keras.Input([96, 96, 1])
    output_tensors = model(input_layer)
    model_output = tf.keras.Model(input_layer, output_tensors)
    return model_output


if __name__ == '__main__':
    model = create_gan_model()