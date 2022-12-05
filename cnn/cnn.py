# %%
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import pandas as pd
import uproot
import time
import math
import cProfile, pstats, io
from pstats import SortKey
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import onnx
import tf2onnx.convert
import onnxruntime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint

import wandb
from wandb.keras import WandbCallback
print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")

from wandb_log import *

wandb.init(entity="mabeyer", project="ort", mode="disabled") # 

# %%
x = None
y = None
with uproot.open("E:/git/TMVA_mcbm/data.root") as file:
    x = np.reshape(np.array(file["train"]["time"].array()), (-1, 72, 32, 1))
    y = np.reshape(np.array(file["train"]["tar"].array()), (-1, 72, 32, 1))

x, x_val, y, y_val = train_test_split(x, y, test_size=0.2)
# %%
def get_hit_average():
    @tf.autograph.experimental.do_not_convert
    def hit_average(data, y_pred):
        y_true = data[:,:,:,0]
        nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
        return (K.sum(y_true*y_pred[:,:,:,0])/nofHits)
    return hit_average   

def hit_average(y_true, y_pred):    
    nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
    return (K.sum(y_true[:,:,:,0]*y_pred[:,:,:,0])/nofHits)   

def noise_average(y_true, y_pred, x):
    noise_mask = tf.subtract(tf.cast(tf.greater(x[:,:,:,0], 0.01), tf.float32), y_true[:,:,:,0])
    nofNoise = tf.math.count_nonzero(tf.greater(noise_mask, 0.01), dtype=tf.float32)
    return (K.sum(noise_mask*y_pred[:,:,:,0])/nofNoise)
# %%
# U-Net

# use_residuals = True

# inputs = Input((72, 32, 1))

# c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (inputs)
# c1 = BatchNormalization()(c1)
# p1 = MaxPooling2D((2, 2)) (c1)

# c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
# c2 = BatchNormalization()(c2)
# p2 = MaxPooling2D((2, 2)) (c2)

# c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
# c3 = BatchNormalization() (c3)
# p3 = MaxPooling2D((2, 2)) (c3)

# mid = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
# u10 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (mid)
# c10 = concatenate([u10, c3], axis=3) if use_residuals else u10

# u10 = Conv2D(64, (3, 3), activation='relu', padding='same') (c10)
# #u11 = UpSampling2D(size=(2, 2))(u10)
# u11 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (u10)
# c11 = concatenate([u11, c2], axis=3) if use_residuals else u11

# u11 = Conv2D(32, (3, 3), activation='relu', padding='same') (c11)
# #u12 = UpSampling2D(size=(2, 2))(u11)
# u12 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (u11)
# c12 = concatenate([u12, c1], axis=3) if use_residuals else u12


# u12 = Conv2D(16, (3, 3), activation='relu', padding='same') (u12)
# outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same') (u12)
# model = Model(inputs=[inputs], outputs=[outputs])
# model.summary()

# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
# cp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# model.compile(optimizer=opt, loss='bce', metrics=[get_hit_average()])

# model.fit(  x=x, y=y,
#             batch_size=128,
#             epochs=1,
#             validation_data=(x_val, y_val),
#             shuffle=True,
#             callbacks=[es, cp_save, WandbCallback()])
# %%
# model = Sequential()
# model.add(Input(shape=(72, 32, 1)))

# model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')) #, strides=2 
# #model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))

# # model.add(MaxPooling2D((2, 2)))
# # model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
# # model.add(UpSampling2D(size=(2, 2)))

# model.add(Conv2D(filters=64 , kernel_size=3, activation='relu', padding='same'))
# model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(filters=32 , kernel_size=3, activation='relu', padding='same'))
# #model.add(UpSampling2D(size=(2, 2)))
# model.add(Conv2D(filters=16 , kernel_size=3, activation='relu', padding='same'))

# model.add(Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))
# model.summary()

# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')
# cp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# model.compile(optimizer=opt, loss='bce', metrics=[get_hit_average()])

# model.fit(  x=x, y=y,
#             batch_size=128,
#             epochs=200,
#             validation_data=(x_val, y_val),
#             shuffle=True,
#             callbacks=[es, cp_save, WandbCallback()])
# %%
model = Sequential()
model.add(InputLayer(input_shape=(72, 32, 1)))
f = 19
model.add(Conv2D(filters=8, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='tanh'))#TODO: try ks=1
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')
cp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

model.compile(optimizer=opt, loss='mse', metrics=[get_hit_average()])

model.fit(  x=x, y=y,
            batch_size=128,
            epochs=100,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=[es, cp_save, WandbCallback()])

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, './mixed.onnx')

outtest = model.predict(x_val[0:1000])
np.savetxt("in_mixed.csv", tf.reshape(x_val[0:1000,:,:,0], [1000, -1]), delimiter=",")
np.savetxt("out_mixed.csv", tf.reshape(outtest[0:1000,:,:,0], [1000, -1]), delimiter=",")

input = (pd.read_csv('./in_mixed.csv', header=None, delimiter= ",", nrows=1000).values.astype(np.float32)).reshape([1000, 72, 32, 1])
# keras_output = (pd.read_csv('./out_mixed.csv', header=None, delimiter= ",", nrows=1000).values.astype(np.float32)).reshape([1000, 72, 32, 1])

ort_sess = onnxruntime.InferenceSession('./mixed.onnx')
ort_sess.get_providers()
ort_sess.set_providers(['CPUExecutionProvider'])
input_name = ort_sess.get_inputs()[0].name
input_shape = ort_sess.get_inputs()[0].shape
batch_size = 1

pr = cProfile.Profile()
pr.enable()

for i in range(1000):
    single_input = input[i][tf.newaxis, ...]
    with tf.device('/cpu:0'):
        model.predict_on_batch(single_input)
    results = ort_sess.run(None, {input_name: single_input})[0]

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
filename = 'profile.prof'
pr.dump_stats(filename)
print(s.getvalue())


# %%
# model.load_weights('model.h5')
# del x, y

# x_test = None
# y_test = None
# with uproot.open("E:/git/TMVA_mcbm/data_test.root") as file:
#     x_test = np.reshape(np.array(file["train"]["time"].array()), (-1, 72, 32, 1))
#     y_test = np.reshape(np.array(file["train"]["tar"].array()), (-1, 72, 32, 1))
# #     x_test = x_test[0:10000]
# #     y_test = y_test[0:10000]

# mask = tf.greater(x_test, 0.001)

# pred_arr = np.empty((0))
# y_test_arr = np.empty((0))
# print("start test iteration")
# for i in range(int(len(x_test)/10000)):
#     x_i = x_test[i*10000:(i+1)*10000]
#     mask_i = tf.greater(x_i, 0.001)
#     pred_i = model.predict(x_i, batch_size=64)
#     pred_i_arr = tf.boolean_mask(pred_i, mask_i).numpy()
#     pred_arr = np.append(pred_arr, pred_i_arr)
#     y_test_i = tf.boolean_mask(y_test[i*10000:(i+1)*10000], mask_i).numpy()
#     y_test_arr = np.append(y_test_arr, y_test_i)
#     print(i)


# LogWandb(y_test_arr, pred_arr)

#pred_test = model.predict(x_test, batch_size=20)
#y_test_arr = tf.boolean_mask(y_test, mask).numpy()
#pred_test_arr = tf.boolean_mask(pred_test, mask).numpy()
#TODO: wandb log roc, auc, cm matrix



# hit_avg_val = 0
# noise_avg_val = 0
# for i in range(692):
#     hit_avg_val += hit_average(y_test[i*128:(i+1)*128], pred_test[i*128:(i+1)*128])
#     noise_avg_val += noise_average(y_test[i*128:(i+1)*128], pred_test[i*128:(i+1)*128], x_test[i*128:(i+1)*128])
# hit_avg_val = hit_avg_val.numpy()/692.0
# noise_avg_val = noise_avg_val.numpy()/692.0
# wandb.log({ "hit_avg_val": hit_avg_val, "noise_avg_val": noise_avg_val })


# %%
# def meanSigma(data):
#     n = len(data)
#     mean = sum(data)/n
#     dev = [(x - mean)**2 for x in data]
#     sigma = math.sqrt(sum(dev)/n)
#     return mean*1e3, sigma*1e3

# pr = cProfile.Profile()
# pr.enable()

# times_cpu = []
# with tf.device('/cpu:0'):
#     for i in range(100):
#         s = x_test[i,:, :, :][tf.newaxis, ...]
#         start = time.process_time()
#         model.predict(s, batch_size=1)
#         #print(time.process_time() - start)
#         stop = time.process_time() - start
#         times_cpu.append(stop)
#     mean, sigma = meanSigma(np.asarray(times_cpu))
#     print(f'mean: {mean:.4f}ms, sigma: {sigma:.4f}ms')
#     wandb.log({"cpu_time_mean": mean})
#     wandb.log({"cpu_time_sigma": sigma})

# times_cpux = []
# with tf.device('/cpu:0'):
#     for i in range(100):
#         s = x_test[i,:, :, :][tf.newaxis, ...]
#         start = time.process_time()
#         model.predict_on_batch(s)
#         #print(time.process_time() - start)
#         stop = time.process_time() - start
#         times_cpux.append(stop)
#     mean, sigma = meanSigma(np.asarray(times_cpux))
#     print(f'xmean: {mean:.4f}ms, xsigma: {sigma:.4f}ms')
#     wandb.log({"cpu_time_xmean": mean})
#     wandb.log({"cpu_time_xsigma": sigma})

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# #ps.print_stats()
# filename = 'profile.prof'
# pr.dump_stats(filename)
# print(s.getvalue())
# times_gpu = []
# with tf.device('/gpu:0'):
#     for i in range(100):
#         s = x_test[i,:, :, :][tf.newaxis, ...]
#         start = time.process_time()
#         model.predict(s, batch_size=1)
#         #print(time.process_time() - start)
#         stop = time.process_time() - start
#         times_gpu.append(stop)
#     mean, sigma = meanSigma(np.asarray(times_gpu))
#     print(f'gpu mean: {mean:.4f}ms, sigma: {sigma:.4f}ms')
#     wandb.log({"gpu_time_mean": mean})
#     wandb.log({"gpu_time_sigma": sigma})
# %%
