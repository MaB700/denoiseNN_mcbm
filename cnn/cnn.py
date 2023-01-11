# %%
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
import time
import cProfile, pstats, io
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")

import onnx
import tf2onnx.convert
import onnxruntime as ort

import helpers
import networks

import wandb
from wandb.keras import WandbCallback
wandb.init(entity="mabeyer", project="ort") # , mode="disabled"

# %%
train_events = 1000 # None for all events
test_events = 32

x, y = None, None
with uproot.open("../data.root") as file:
    x = np.reshape(np.array(file["train"]["time"].array(entry_stop=train_events)), (-1, 72, 32, 1))
    y = np.reshape(np.array(file["train"]["tar"].array(entry_stop=train_events)), (-1, 72, 32, 1))

x, x_val, y, y_val = train_test_split(x, y, test_size=0.2)

# %%
# model = networks.Stacked()
# model.summary()
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=opt, loss='bce', metrics=[helpers.get_hit_average()])



model = networks.UNet()
model.summary()

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')
cp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
model.fit(  x=x, y=y,
            batch_size=256,
            epochs=0,
            validation_data=(x_val, y_val),
            callbacks=[es, cp_save, WandbCallback()])
# %%
del x, y
# model.load_weights('model.h5')

# x_test = None
# y_test = None
# with uproot.open("../data_test.root") as file:
#     x_test = np.reshape(np.array(file["train"]["time"].array(entry_stop=test_events)), (-1, 72, 32, 1))
#     y_test = np.reshape(np.array(file["train"]["tar"].array(entry_stop=test_events)), (-1, 72, 32, 1))

# pred = model.predict(x_test, batch_size=32)
# # plot pred in a 2d colour plot
# import matplotlib.pyplot as plt

# plt.figure(figsize=(20, 20))
# for i in range(1, 4):
#     ax = plt.subplot(2, 2, i)
#     plt.imshow(pred[i, :, :, 0], interpolation='none', cmap='gray')
#     plt.title(str(i))
#     plt.colorbar()
# plt.show(block=True)



# pred_arr = np.empty((0))
# y_test_arr = np.empty((0))
# hit_avg_val = 0
# noise_avg_val = 0
# print("start test iteration")
# for i in range(int(len(x_test)/10000)): # predict in batches of 10k
#     x_i = x_test[i*10000:(i+1)*10000]
#     mask_i = tf.greater(x_i, 0.01)
#     pred_i = model.predict(x_i, batch_size=128)
#     pred_i_arr = tf.boolean_mask(pred_i, mask_i).eval(session=tf.compat.v1.Session())
#     pred_arr = np.append(pred_arr, pred_i_arr)
#     y_test_i = tf.boolean_mask(y_test[i*10000:(i+1)*10000], mask_i).eval(session=tf.compat.v1.Session())
#     y_test_arr = np.append(y_test_arr, y_test_i)
#     hit_avg_val += helpers.hit_average(y_test[i*10000:(i+1)*10000], pred_i)
#     noise_avg_val += helpers.noise_average(y_test[i*10000:(i+1)*10000], pred_i, x_i)
#     print("test iteration {} of {}".format(i+1, int(len(x_test)/10000)))

# helpers.LogWandb(y_test_arr, pred_arr)
# wandb.log({ "hit_avg_val": hit_avg_val, "noise_avg_val": noise_avg_val })

# %%
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, './unet.onnx')

# options = ort.SessionOptions()
# # options.execution_mode.ORT_PARALLEL
# sess = ort.InferenceSession('./model.onnx', providers=['CPUExecutionProvider'], sess_options=options)
# input_name = sess.get_inputs()[0].name
# input_shape = sess.get_inputs()[0].shape
# pr1 = cProfile.Profile()
# pr1.enable()
# for i in range(10000):
#     input_i = x_test[i][tf.newaxis, ...]
#     sess.run(None, {input_name: input_i})[0]
# pr1.disable()
# s1 = io.StringIO()
# ps1 = pstats.Stats(pr1, stream=s1)
# latency_s = ps1.get_stats_profile().func_profiles['run'].cumtime / 10.0
# wandb.log({ "latency_s [ms]": latency_s})

# options.execution_mode.ORT_PARALLEL
# sess = ort.InferenceSession('./model.onnx', providers=['CPUExecutionProvider'], sess_options=options)
# pr2 = cProfile.Profile()
# pr2.enable()
# for i in range(10000):
#     input_i = x_test[i][tf.newaxis, ...]
#     sess.run(None, {input_name: input_i})[0]
# s2 = io.StringIO()
# ps2 = pstats.Stats(pr2, stream=s2)
# latency_p = ps2.get_stats_profile().func_profiles['run'].cumtime /10.0
# wandb.log({ "latency_p [ms]": latency_p})

# with tf.device('/cpu:0'):
#     outtest = model.predict(x_val[0:100])
# np.savetxt("in_mixed.csv", tf.reshape(x_val[0:100,:,:,0], [100, -1]), delimiter=",")
# np.savetxt("out_mixed.csv", tf.reshape(outtest[0:100,:,:,0], [100, -1]), delimiter=",")