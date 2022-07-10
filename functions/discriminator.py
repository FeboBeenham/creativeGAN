import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_critic(height, width, depth, alpha=0.2):
    # create a Keras Sequential model
    model = Sequential(name="critic")
    input_shape = (height, width, depth)

    # 1. first set of CONV => BN => leaky ReLU layers
    model.add(layers.Conv2D(128, (4, 4),
                                padding="same",
                                strides=(2, 2),
                                input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=alpha))

    # 2. second set of CONV => BN => leacy ReLU layers
    model.add(layers.Conv2D(256, (4, 4),
                                padding="same",
                                strides=(2, 2)))
    model.add(layers.LeakyReLU(alpha=alpha))

    # 3. third set of CONV => BN => leacy ReLU layers
    model.add(layers.Conv2D(512, (4, 4),
                                padding="same",
                                strides=(2, 2)))
    model.add(layers.LeakyReLU(alpha=alpha))

    # 3. third set of CONV => BN => leacy ReLU layers
    model.add(layers.Conv2D(1024, (4, 4),
                                padding="same",
                                strides=(2, 2)))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Dropout(0.3))


    # flatten and apply dropout
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation="linear"))

    # return the critic model
    return model
