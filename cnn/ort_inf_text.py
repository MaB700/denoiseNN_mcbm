import numpy as np
import pandas as pd
import time
import cProfile, pstats, io
from pstats import SortKey
import onnxruntime as ort

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.execution_mode.ORT_PARALLEL
sess = ort.InferenceSession('./Linear_16x.onnx', providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

input = np.random.rand(16, 100).astype(np.float32)
pr = cProfile.Profile()
pr.enable()
for i in range(10000):
    ort_inf = sess.run(None, {input_name: input})[0]

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
filename = 'profile.prof'
pr.dump_stats(filename)
print(s.getvalue())
t = ps.get_stats_profile().func_profiles['run'].cumtime / 10000 * 1000**2 #time in mikro seconds
print(f'Average time per run in mikro seconds: {t:.3f}')

