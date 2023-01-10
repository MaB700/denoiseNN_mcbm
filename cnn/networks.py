import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# import tensorflow._api.v2.compat.v1 as tf

# tf.disable_v2_behavior()
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def Stacked():
    model = Sequential()
    model.add(InputLayer(input_shape=(72, 32, 1)))
    model.add(Conv2D(filters=8, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
    return model

def custom_loss(i):
    i = ops.convert_to_tensor_v2_with_dispatch(i)
    mask = K.greater(i, 0.01)
    def loss(y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(K.binary_crossentropy(y_true[mask], y_pred[mask]), axis=-1)
    return loss

def Stacked2():
    i = Input((72, 32, 1))
    c1 = Conv2D(8, (5, 5), activation='relu', padding='same') (i)
    c2 = Conv2D(16, (5, 5), activation='relu', padding='same') (c1)
    c3 = Conv2D(32, (5, 5), activation='relu', padding='same') (c2)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c4)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same') (c5)
    o = Conv2D(1, (3, 3), activation='sigmoid', padding='same') (c6)
    model = Model(i, o)
    model.compile(optimizer='adam', loss=custom_loss(i), metrics=['accuracy'], experimental_run_tf_function=False)
    return model

def TestNet():
    i = Input((72, 32, 1))
    # do pointwise convolution
    c1 = Conv2D(8, (1, 1), activation='relu', padding='same') (i)
    # do depthwise convolution
    c2 = Conv2D(8, (5, 5), activation='relu', padding='same', use_bias=False, depthwise_regularizer='l2') (c1)

def UNet(use_residuals = True):
    i = Input((72, 32, 1))

    c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (i)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
    c3 = BatchNormalization() (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    mid = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
    u10 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (mid)
    c10 = concatenate([u10, c3], axis=3) if use_residuals else u10

    u10 = Conv2D(64, (3, 3), activation='relu', padding='same') (c10)
    #u11 = UpSampling2D(size=(2, 2))(u10)
    u11 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (u10)
    c11 = concatenate([u11, c2], axis=3) if use_residuals else u11

    u11 = Conv2D(32, (3, 3), activation='relu', padding='same') (c11)
    #u12 = UpSampling2D(size=(2, 2))(u11)
    u12 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (u11)
    c12 = concatenate([u12, c1], axis=3) if use_residuals else u12

    u13 = Conv2D(16, (3, 3), activation='relu', padding='same') (c12)
    c13 = concatenate([u13, i], axis=3) if use_residuals else u13
    o = Conv2D(1, (3, 3), activation='sigmoid', padding='same') (c13)
    model = Model(i, o) #custom_loss(i)
    model.compile(optimizer='adam', loss=custom_loss(i), metrics=['accuracy'], experimental_run_tf_function=False)
    return model

def Autoencoder():
    model = Sequential()
    model.add(Input(shape=(72, 32, 1)))

    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))

    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    # model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(filters=64 , kernel_size=3, activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=32 , kernel_size=3, activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=16 , kernel_size=3, activation='relu', padding='same'))

    model.add(Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))
    return model

class Stacked0(Model):
    # Subclassing does not work properly with file saving and FLOPS calculation of Wandb
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(8, kernel_size=5, padding='same', activation='relu')
        self.conv2 = Conv2D(16, kernel_size=5, padding='same', activation='relu')
        self.conv3 = Conv2D(32, kernel_size=5, padding='same', activation='relu')
        self.conv4 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv5 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv6 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv_out = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.conv_out(x)

    def summary(self):
        x = Input(shape=(72, 32, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()