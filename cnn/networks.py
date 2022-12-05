from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate

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

def UNet(use_residuals = True):
    inputs = Input((72, 32, 1))

    c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (inputs)
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

    u12 = Conv2D(16, (3, 3), activation='relu', padding='same') (u12)
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same') (u12)
    return Model(inputs=[inputs], outputs=[outputs])

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