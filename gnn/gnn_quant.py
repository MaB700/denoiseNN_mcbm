import time
from helpers_custom import *
from torch_geometric.data.data import Data
import onnxruntime as ort
import onnxruntime.quantization as oq

model = customGNN()

data = Data(x=torch.randn(25, 3), edge_index=torch.randint(0, 25, (2, 40)))
ONNX_FILE_PATH = "customgnn.onnx"
dynamic_axes = {"nodes": {0: "num_nodes", 1:"node_features"}, "edge_index": {1: "num_edges"}, "output": {0: "num_nodes"}}
torch.onnx.export(model, (data.x, data.edge_index), ONNX_FILE_PATH, input_names=["nodes", "edge_index"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)

expected = model(data.x, data.edge_index)
start = time.time()
for i in range(1000):
    expected = model(data.x, data.edge_index)
end = time.time()
print("Average time of inference pytorch: ", (end - start) / 1000 * 1000, "ms")
options = ort.SessionOptions()
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# options.intra_op_num_threads = 4 # athena uses 1
# options.inter_op_num_threads = 4
# options.enable_profiling=True
# options.execution_mode.ORT_PARALLEL
session = ort.InferenceSession(ONNX_FILE_PATH, options)
out = session.run(None, {"nodes": data.x.numpy(), "edge_index": data.edge_index.numpy()})[0]
torch.testing.assert_close(expected.detach().numpy(), out, rtol=1e-03, atol=1e-05)

x = data.x.numpy()
ei = data.edge_index.numpy()
start = time.time()
for i in range(1000):
    y = session.run(None, {"nodes": x, "edge_index": ei})[0]
end = time.time()
print("Average time of inference ort_fp32: ", (end - start) / 1000 * 1000, "ms")





# # FLOAT16, SCATTER ELEMENTS OPSET 16 'add' NOT SUPPORTED BY ONNXRUNTIME
# import onnxmltools
# from onnxmltools.utils.float16_converter import convert_float_to_float16

# input_onnx_model = './customgnn.onnx'
# output_onnx_model = './customgnn_fp16.onnx'

# onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# onnx_model = convert_float_to_float16(onnx_model)
# onnxmltools.utils.save_model(onnx_model, output_onnx_model)
# session2 = ort.InferenceSession('./customgnn_fp16.onnx', options)
# x = data.x.numpy().astype(np.float16)
# ei = data.edge_index.numpy()
# start = time.time()
# for i in range(1000):
#     y = session2.run(None, {"nodes": x, "edge_index": ei})[0]
# end = time.time()
# print("Average time of inference ort_fp16: ", (end - start) / 1000 * 1000, "ms")


# INT8 quantization
# from onnxruntime.quantization.calibrate import CalibrationDataReader

# class CalibrationDataProvider(CalibrationDataReader):
#     def __init__(self):
#         super(CalibrationDataProvider, self).__init__()
#         self.counter = 0
#         self.x = torch.randn(1000, 25, 3).numpy()
#         self.edge_index = torch.randint(0, 25, (1000, 2, 40)).numpy()
        

#     def get_next(self):
#         if self.counter > 1000 - 1:
#             return None
#         else:
#             out = {'nodes': self.x[self.counter], 'edge_index': self.edge_index[self.counter]}
#             self.counter += 1
#             return out

# cdp = CalibrationDataProvider()
# quantized_onnx_model = oq.quantize_static('./customgnn.onnx', './customgnn_quant.onnx', weight_type=oq.QuantType.QInt8, calibration_data_reader=cdp, per_channel=True, reduce_range=True)

# session2 = ort.InferenceSession('./customgnn_quant.onnx', options)
# start = time.time()
# for i in range(1000):
#     x = session2.run(None, {"nodes": data.x.numpy(), "edge_index": data.edge_index.numpy()})[0]
# end = time.time()
# print("Average time of inference ort_quant: ", (end - start) / 1000 * 1000, "ms")
