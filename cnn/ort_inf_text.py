import os
import numpy as np
import time
import cProfile, pstats, io
from pstats import SortKey
import onnxruntime as ort
import olive
from olive.optimization_config import OptimizationConfig
from olive.optimize import optimize

# import openvino.utils as utils

def modelopt():
    opt_config = OptimizationConfig(
        model_path = "lin5.onnx",
        result_path = "olive_opt_latency_result",
        providers_list = ["cpu","cuda"], # ,"dnnl", "openvino"
        #inter_thread_num_list = [1,4],
        #intra_thread_num_list = [1,4],
        #execution_mode_list = ["sequential", "parallel"],
        ort_opt_level_list = ["all"],
        warmup_num = 1000,
        test_num = 10000)

    # opt_config = OptimizationConfig(
    #     model_path = "Linear_16x.onnx",
    #     result_path = "olive_opt_throughput_result",
    #     throughput_tuning_enabled=True,
    #     inputs_spec = {"input_1": [-1, 100]},
    #     max_latency_percentile = 0.95,
    #     max_latency_ms = 100,
    #     threads_num = 1,
    #     dynamic_batching_size = 4,
    #     min_duration_sec=10)

    result = optimize(opt_config)

if __name__ == "__main__":
    # utils.add_openvino_libs_to_path()
    modelopt()



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

