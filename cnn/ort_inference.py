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
sess.get_providers()
sess.set_providers(['CPUExecutionProvider'])

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
batch_size = 1

input = input.reshape([100, 72, 32, 1])
keras_output = keras_output.reshape([100, 72, 32, 1])

# start = time.process_time()
pr = cProfile.Profile()
pr.enable()
ort_inf = np.empty((100, 72, 32, 1))
for i in range(100):
    input_i = input[i][tf.newaxis, ...]
    ort_inf[i] = sess.run(None, {input_name: input_i})[0]
    
print(ort_inf.shape)
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
hit_indices = input > 0.01
np.testing.assert_allclose(ort_inf[hit_indices], keras_output[hit_indices], atol=1e-7, rtol=1e-5, verbose=True)

# for i in range(10):
#     print('keras: {}  ort: {}'.format(keras_output[hit_indices][i], ort_inf[hit_indices][i]))