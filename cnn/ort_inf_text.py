import os
import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
import onnxruntime as ort
import olive
from olive.optimization_config import OptimizationConfig
from olive.optimize import optimize

opt_config = OptimizationConfig(
    model_path = "./mixed.onnx",
    result_path = "olive_opt_latency_result",
    providers_list = ["cpu"],
    inter_thread_num_list = [1,2,4, 8],
    execution_mode_list = ["parallel", "sequential"],
    warmup_num = 10,
    test_num = 40)

result = optimize(opt_config)



# options = ort.SessionOptions()
# options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# options.execution_mode.ORT_PARALLEL
# sess = ort.InferenceSession('./mixed.onnx', providers=['CPUExecutionProvider'])
# input_name = sess.get_inputs()[0].name
# input_shape = sess.get_inputs()[0].shape

# input = np.random.rand(1, 72, 32, 1).astype(np.float32)
# pr = cProfile.Profile()
# pr.enable()
# for i in range(10000):
#     ort_inf = sess.run(None, {input_name: input})[0]

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# filename = 'profile.prof'
# pr.dump_stats(filename)
# print(s.getvalue())
# t = ps.get_stats_profile().func_profiles['run'].cumtime / 10000 * 1000**2 #time in mikro seconds
# print(f'Average time per run in mikro seconds: {t:.3f}')

