from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     Conv2DTranspose, concatenate, Input)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3


def build_model(start_neurons, conv_kernel_size, activation_type: str, 
                learning_rate):
    activation_type = str(activation_type)

    input_layer = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv1 = Conv2D(start_neurons * 1, conv_kernel_size,
                   activation=activation_type, padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, conv_kernel_size,
                   activation=activation_type, padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, conv_kernel_size,
                   activation=activation_type, padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, conv_kernel_size,
                   activation=activation_type, padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, conv_kernel_size,
                   activation=activation_type, padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, conv_kernel_size,
                   activation=activation_type, padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, conv_kernel_size,
                   activation=activation_type, padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, conv_kernel_size,
                   activation=activation_type, padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, conv_kernel_size,
                   activation=activation_type, padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, conv_kernel_size,
                   activation=activation_type, padding="same")(convm)

    deconv4 = Conv2DTranspose(start_neurons * 8, conv_kernel_size,
                              strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, conv_kernel_size,
                              strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, conv_kernel_size,
                              strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, conv_kernel_size,
                              strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, conv_kernel_size,
                    activation=activation_type, padding="same")(uconv1)

    output_layer = Conv2D(3, (1, 1), padding="same",
                          activation="softmax")(uconv1)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
