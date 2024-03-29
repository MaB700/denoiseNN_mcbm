# import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer


import wandb
from wandb.keras import WandbCallback
wandb.init(entity="mabeyer", project="ort", mode='disabled')


x = np.random.rand(1000, 100)
y = np.random.rand(1000, 50)

model = Sequential()
model.add(InputLayer(input_shape=(1,)))
for _ in range(100):
    model.add(Dense(1, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(  x=x, y=y,
            batch_size=16,
            epochs=0,
            callbacks=[WandbCallback()])

# save model to onnx file format
import onnx
import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, './lin5.onnx')


