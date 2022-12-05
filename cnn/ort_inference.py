import numpy as np
import pandas as pd
import time
import cProfile, pstats, io
from pstats import SortKey
import tensorflow as tf
import onnxruntime as ort

input = pd.read_csv('./in_mixed.csv', header=None, delimiter= ",", nrows=100).values.astype(np.float32)
keras_output = pd.read_csv('./out_mixed.csv', header=None, delimiter= ",", nrows=100).values.astype(np.float32)

sess = ort.InferenceSession('./mixed.onnx')

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
batch_size = 1

input = input.reshape([100, 72, 32, 1])
keras_output = keras_output.reshape([100, 72, 32, 1])

# start = time.process_time()
pr = cProfile.Profile()
pr.enable()

for i in range(100):
    ort_inf = sess.run(None, {input_name: input})[0]

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
filename = 'profile.prof'
pr.dump_stats(filename)
print(s.getvalue())

# delta_t = time.process_time() - start
# print(delta_t)

avg_absolute = np.average(np.abs(np.subtract(keras_output, ort_inf)))
avg_relative = np.average(np.abs(np.divide(np.subtract(keras_output, ort_inf), keras_output)))
print('Average absolut deviation: {}'.format(avg_absolute))
print('Average relative deviation: {}'.format(avg_relative))

max_absolute = np.amax(np.abs(np.subtract(keras_output, ort_inf)))
max_relative = np.amax(np.abs(np.divide(np.subtract(keras_output, ort_inf), keras_output)))
print('Max. absolut deviation: {}'.format(max_absolute))
print('Max. relative deviation: {}'.format(max_relative))

hit_indices = input > 0.01 # smalles values is always 1/26 ~ 0.0385
max_hit_absolute = np.amax(np.abs(np.subtract(keras_output[hit_indices], ort_inf[hit_indices])))
max_hit_relative = np.amax(np.abs(np.divide(np.subtract(keras_output[hit_indices], ort_inf[hit_indices]), keras_output[hit_indices])))
print('Max. absolut deviation hits: {}'.format(max_hit_absolute))
print('Max. relative deviation hits: {}'.format(max_hit_relative))
