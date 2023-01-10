import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
#import tensorflow as tf
import onnxruntime as ort
# import openvino.utils as utils 
# utils.add_openvino_libs_to_path()

# input = pd.read_csv('./in_mixed.csv', header=None, delimiter= ",", nrows=100).values.astype(np.float32)
# keras_output = pd.read_csv('./out_mixed.csv', header=None, delimiter= ",", nrows=100).values.astype(np.float32)

# use onnxruntime SetGlobalDenormalAsZero to speed up inference

device = 'CPU_FP32'
options = ort.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# options.intra_op_num_threads = 1
# options.enable_profiling=True
# options.execution_mode.ORT_PARALLEL
# sess = ort.InferenceSession('./model.onnx', providers=['OpenVINOExecutionProvider'], sess_options=options, device_id=0, device_type=device)
sess = ort.InferenceSession('../cnn_torch/model.onnx', providers=['CPUExecutionProvider'], sess_options=options, device_id=0, device_type=device)

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
batch_size = 1

# input = input.reshape([100, 72, 32, 1])
# keras_output = keras_output.reshape([100, 72, 32, 1])
b = 8
input = np.random.random((b, 1, 72, 32)).astype(np.float32)

pr = cProfile.Profile()
#pr.enable()
# start = time.process_time()
ort_inf = np.empty((100, 72, 32, 1))
for i in range(1010):
    #input_i = input[i % 100][tf.newaxis, ...]
    if i==10: pr.enable()
    sess.run(None, {input_name: input})[0]


# delta_t = time.process_time() - start
# print('time in s: {}'.format(delta_t/100))
#print(ort_inf.shape)
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
print(ps.get_stats_profile().func_profiles['run'].cumtime / (b*1000) * 1000) # in ms
ps.print_stats()
filename = 'profile.prof'
pr.dump_stats(filename)
print(s.getvalue())
# write the profile to a numpy array



# np.testing.assert_allclose(ort_inf, keras_output, atol=3e-7, rtol=1e-5, verbose=True)
# hit_indices = input > 0.01
# np.testing.assert_allclose(ort_inf[hit_indices], keras_output[hit_indices], atol=1e-7, rtol=1e-5, verbose=True)

# for i in range(10):
#     print('keras: {}  ort: {}'.format(keras_output[hit_indices][i], ort_inf[hit_indices][i]))