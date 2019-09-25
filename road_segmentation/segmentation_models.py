import tensorflow as tf
from tensorflow.keras import layers, models


def simple_model(input_shape):
    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv


def de_conv2d_3x3(filters):
    return layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same')


def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')


def unet(input_shape):
    image = layers.Input(shape=input_shape)

    def contract(input_filters, input_feature_map):
        conv = conv2d_3x3(input_filters)(input_feature_map)
        conv = conv2d_3x3(input_filters)(conv)
        mp = max_pool()(conv)

        return mp, conv

    def expand(middle_filters, output_filters, input_feature_map):
        conv = conv2d_3x3(middle_filters)(input_feature_map)
        conv = conv2d_3x3(middle_filters)(conv)
        de_conv = de_conv2d_3x3(output_filters)(conv)

        return de_conv

    def output( middle_filters, input_feature_map):
        conv = conv2d_3x3(middle_filters)(input_feature_map)
        conv = conv2d_3x3(middle_filters)(conv)
        probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(conv)
        return probs

    # Contraction(c1 = contraction_x, mfm1 = multichannel_feature_map_x
    c1, mfm1 = contract(input_filters=64, input_feature_map=image)
    c2, mfm2 = contract(input_filters=128, input_feature_map=c1)
    c3, mfm3 = contract(input_filters=256, input_feature_map=c2)
    c4, mfm4 = contract(input_filters=512, input_feature_map=c3)

    # Bottleneck
    bn = expand(middle_filters=1025, output_filters=512, input_feature_map=c4)

    # Expansion
    c4_bn_cat = tf.concat([mfm4, bn], -1)
    e1 = expand(middle_filters=512, output_filters=256, input_feature_map=c4_bn_cat)
    c3_e1_cat = tf.concat([mfm3, e1], -1)
    e2 = expand(middle_filters=256, output_filters=128, input_feature_map=c3_e1_cat)
    c2_e2_cat = tf.concat([mfm2, e2], -1)
    e3 = expand(middle_filters=128, output_filters=64, input_feature_map=c2_e2_cat)

    # Output
    c1_e3_cat = tf.concat([mfm1, e3], -1)
    probs = output(middle_filters=64, input_feature_map=c1_e3_cat)

    model = models.Model(inputs=image, outputs=probs)

    return model
