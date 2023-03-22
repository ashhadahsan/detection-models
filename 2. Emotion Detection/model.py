from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Dropout, BatchNormalization

img_width, img_height, img_depth = 48, 48, 1


def create_model():
    def build_net(optim):
        """
        This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
        I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
        atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
        results.
        """
        net = Sequential(name="DCNN")

        net.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                input_shape=(img_width, img_height, img_depth),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_1",
            )
        )
        net.add(BatchNormalization(name="batchnorm_1"))
        net.add(
            Conv2D(
                filters=64,
                kernel_size=(5, 5),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_2",
            )
        )
        net.add(BatchNormalization(name="batchnorm_2"))

        net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
        net.add(Dropout(0.4, name="dropout_1"))

        net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_3",
            )
        )
        net.add(BatchNormalization(name="batchnorm_3"))
        net.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_4",
            )
        )
        net.add(BatchNormalization(name="batchnorm_4"))

        net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"))
        net.add(Dropout(0.4, name="dropout_2"))

        net.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_5",
            )
        )
        net.add(BatchNormalization(name="batchnorm_5"))
        net.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                activation="elu",
                padding="same",
                kernel_initializer="he_normal",
                name="conv2d_6",
            )
        )
        net.add(BatchNormalization(name="batchnorm_6"))

        net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"))
        net.add(Dropout(0.5, name="dropout_3"))

        net.add(Flatten(name="flatten"))

        net.add(
            Dense(128, activation="elu", kernel_initializer="he_normal", name="dense_1")
        )
        net.add(BatchNormalization(name="batchnorm_7"))

        net.add(Dropout(0.6, name="dropout_4"))

        net.add(Dense(num_classes, activation="softmax", name="out_layer"))

        net.compile(
            loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]
        )

        net.summary()

        return net
