# %%
#load modules
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import fixed
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #uncomment to use cpu
import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
from tensorflow.keras.layers import InputLayer, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential

# load custom functions/loss/metrics
from denoiseNN_functions import *

#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="autoencoder_mcbm_toy_denoise")

print('Tensorflow version: ' + tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")
# %%
#load dataset
#header gives information of parameters
#TODO: load parameters from file header
nofEvents_train = 20000
nofEvents_test = 3213
cut_range = 20.0
px_x = 32
px_y = 72

with open("E:/ML_data/mcbm_rich/28.07/hits_all.txt", 'r') as temp_f:
    col_count1 = [ len(l.split(",")) for l in temp_f.readlines() ]
column_names1 = [i for i in range(0, max(col_count1))]

with open("E:/ML_data/mcbm_rich/28.07/hits_true.txt", 'r') as temp_f:
    col_count2 = [ len(l.split(",")) for l in temp_f.readlines() ]
column_names2 = [i for i in range(0, max(col_count2))]

with open("E:/ML_data/mcbm_rich/28.07/hits_all_test.txt", 'r') as temp_f:
    col_count3 = [ len(l.split(",")) for l in temp_f.readlines() ]
column_names3 = [i for i in range(0, max(col_count3))]

with open("E:/ML_data/mcbm_rich/28.07/hits_true_test.txt", 'r') as temp_f:
    col_count4 = [ len(l.split(",")) for l in temp_f.readlines() ]
column_names4 = [i for i in range(0, max(col_count4))]

hits_all_train = pd.read_csv("E:/ML_data/mcbm_rich/28.07/hits_all.txt",header=None ,index_col=0,comment='#', delimiter=",", names=column_names1).values.astype('int32')
hits_true_train = pd.read_csv("E:/ML_data/mcbm_rich/28.07/hits_true.txt",header=None ,index_col=0,comment='#', delimiter=",", names=column_names2).values.astype('int32')
hits_all_test = pd.read_csv("E:/ML_data/mcbm_rich/28.07/hits_all_test.txt",header=None ,index_col=0,comment='#', delimiter=",", names=column_names3).values.astype('int32')
hits_true_test = pd.read_csv("E:/ML_data/mcbm_rich/28.07/hits_true_test.txt",header=None ,index_col=0,comment='#', delimiter=",", names=column_names4).values.astype('int32')


hits_all_train[hits_all_train < 0] = 0
hits_true_train[hits_true_train < 0] = 0
hits_all_test[hits_all_test < 0] = 0
hits_true_test[hits_true_test < 0] = 0


train = np.zeros([len(hits_all_train[:,0]), px_x*px_y])
train2 = np.zeros([len(hits_true_train[:,0]), px_x*px_y])
test = np.zeros([len(hits_all_test[:,0]), px_x*px_y])
test2 = np.zeros([len(hits_true_test[:,0]), px_x*px_y])

for i in range(len(hits_all_train[:,0])):
    for j in range(len(hits_all_train[0,:])):
        if hits_all_train[i,j]==0:
            break
        train[i,hits_all_train[i,j]-1]+=1

for i in range(len(hits_true_train[:,0])):
    for j in range(len(hits_true_train[0,:])):
        if hits_true_train[i,j]==0:
            break
        train2[i,hits_true_train[i,j]-1]+=1

for i in range(len(hits_all_test[:,0])):
    for j in range(len(hits_all_test[0,:])):
        if hits_all_test[i,j]==0:
            break
        test[i,hits_all_test[i,j]-1]+=1

for i in range(len(hits_true_test[:,0])):
    for j in range(len(hits_true_test[0,:])):
        if hits_true_test[i,j]==0:
            break
        test2[i,hits_true_test[i,j]-1]+=1


train = tf.reshape(train, [len(hits_all_train[:,0]), px_y, px_x])
train2 = tf.reshape(train2, [len(hits_true_train[:,0]), px_y, px_x])
test = tf.reshape(test, [len(hits_all_test[:,0]), px_y, px_x])
test2 = tf.reshape(test2, [len(hits_true_test[:,0]), px_y, px_x])

train = tf.clip_by_value(train, clip_value_min=0., clip_value_max=1.)
train2 = tf.clip_by_value(train2, clip_value_min=0., clip_value_max=1.)
test = tf.clip_by_value(test, clip_value_min=0., clip_value_max=1.)
test2 = tf.clip_by_value(test2, clip_value_min=0., clip_value_max=1.)
#print(test2)
#single_event_plot(test,test,px_x, -8.1, 13.1, px_y, -23.85, 23.85, 7, cut =0 )

#interactive_plot = widgets.interact(single_event_plot, data=fixed(test2), data0=fixed(test), nof_pixel_X=fixed(px_x), min_X=fixed(-13.1), max_X=fixed(8.1), \
                    #nof_pixel_Y=fixed(px_y), min_Y=fixed(-23.85), max_Y=fixed(23.85), eventNo=(0,len(hits_all_test[:,0])-1,1), cut=(0.,0.90,0.05))

# %%
#preprocessing
hits_all_train = tf.cast(train[..., tf.newaxis],dtype=tf.float32)
hits_all_test = tf.cast(test[..., tf.newaxis],dtype=tf.float32)
hits_true_train = tf.cast(train2[..., tf.newaxis],dtype=tf.float32)
hits_true_test = tf.cast(test2[..., tf.newaxis],dtype=tf.float32)

# create only noise
noise_train = tf.clip_by_value(tf.math.add(hits_all_train, -hits_true_train),clip_value_min=0., clip_value_max=1.)
noise_test = tf.clip_by_value(tf.math.add(hits_all_test, -hits_true_test), clip_value_min=0., clip_value_max=1.)

order1_train = create_orderN(noise_train, 1)
order2_train = create_orderN(noise_train, 2)
order2_train -= order1_train
order2_train = tf.clip_by_value(order2_train, clip_value_min=0., clip_value_max=1.)

order1_test = create_orderN(noise_test, 1)
order2_test = create_orderN(noise_test, 2)
order2_test -= order1_test
order2_test = tf.clip_by_value(order2_test, clip_value_min=0., clip_value_max=1.)


hits_train = tf.concat([hits_true_train, noise_train, order1_train, order2_train], 3)
hits_test = tf.concat([hits_true_test, noise_test, order1_test, order2_test], 3)

#del noise_train, noise_test, order1_train, order2_train, order1_test, order2_test #free up momory

# %%
#train nn
custom_metrics = [get_hit_average(), get_noise_average(), get_background_average(),\
                    get_hit_average_order1(), get_hit_average_order2()]

model = Sequential()
model.add(InputLayer(input_shape=(72, 32, 1)))

model.add(Conv2D(filters=32, kernel_size=3, strides=2,activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=[2, 2],activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, strides=[2, 2],activation='relu', padding='same'))

model.add(Conv2DTranspose(filters=128 , kernel_size=3, strides=[2, 2],activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=64 , kernel_size=3, strides=[2, 2],activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=32 , kernel_size=3, strides=2,activation='relu', padding='same'))
model.add(Conv2D(1, kernel_size=3, activation='tanh', padding='same'))
#model.add(Conv2D(1, kernel_size=1, activation=''))
model.summary()
#opt = tf.keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-07 )
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=get_custom_loss(), metrics=custom_metrics)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
model.fit(hits_all_train, hits_train,
                epochs=5,
                batch_size=300,
                shuffle=True,
                validation_data=(hits_all_test, hits_test),
                callbacks=[])#,
                #callbacks=[WandbCallback(log_weights=True)])
#print('model evaluate ...\n')
#model.evaluate(hits_noise_test, hits_test, verbose=1)

original_plt = tf.math.add(hits_test[:,:,:,0], tf.math.scalar_mul(2.0, hits_test[:,:,:,1]) )
encoded = model.predict(hits_all_test, batch_size=50)

# # single_event_plot(hits_noise_train, 48, -20.0, 20.0, 48, -20.0, 20.0, 0)

# %%
interactive_plot = widgets.interact(single_event_plot, \
                    data=fixed(tf.squeeze(encoded,[3])), data0=fixed(2*hits_test[:,:,:,1]+hits_test[:,:,:,0]), \
                    nof_pixel_X=fixed(px_x), min_X=fixed(-8.1), max_X=fixed(13.1), \
                    nof_pixel_Y=fixed(px_y), min_Y=fixed(-23.85), max_Y=fixed(23.85), eventNo=(50,100-1,1), cut=(0.,0.90,0.05))



# # %%
# #model.evaluate((hits_noise_test[14,:,:,:])[tf.newaxis,...], (hits_test[14,:,:,:])[tf.newaxis,...], verbose=1);
# def hit_average_order2(data, y_pred):
#     y_hits_in_order2 = data[14,:,:,0]*data[14,:,:,3]
#     nofHitsInOrder2 = tf.math.count_nonzero(tf.greater(y_hits_in_order2,0.01), dtype=tf.float32)
#     print(nofHitsInOrder2)
#     return (K.sum(y_hits_in_order2*y_pred[14,:,:,0])/nofHitsInOrder2), nofHitsInOrder2

# or2, nofhitsin02 = hit_average_order2(hits_test, encoded )
# print(or2)
# print(nofhitsin02)
# # %%
# print(tf.math.count_nonzero(tf.greater(hits_test[0,:,:,2],0.), dtype=tf.float32))
# #tf.greater(hits_test[0,:,:,2],0.5)
# #tf.print(hits_test[0,:,:,2], summarize=-1)
# #print(hits_test[0,:,:,2])
# # %%

