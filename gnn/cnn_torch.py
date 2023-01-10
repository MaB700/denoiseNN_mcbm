import numpy as np
import torch
from torch.nn import Conv2d, ConvTranspose2d, Linear
import torch.nn as nn

conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3)
params = sum(p.numel() for p in conv.parameters() if p.requires_grad)

x = torch.rand(1, 1, 50, 50)
out = conv(x)

depth_conv = Conv2d(in_channels=1, out_channels=10, kernel_size=3, groups=1)
point_conv = Conv2d(in_channels=10, out_channels=32, kernel_size=1)

depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

out_depthwise = depthwise_separable_conv(x)

print(f"The standard convolution uses {params} parameters.")
print(f"The depthwise separable convolution uses {params_depthwise} parameters.")

assert out.shape == out_depthwise.shape, "Size mismatch"

# from fvcore.nn import FlopCountAnalysis

# flops = FlopCountAnalysis(conv, x)
# print(f"The standard convolution uses {flops.total():,} flops.")

# flops = FlopCountAnalysis(depthwise_separable_conv, x)
# print(f"The depthwise separable convolution uses {flops.total():,} flops.")

c1 = Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1) # replicator
c2 = Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, groups=2) # depthwise
c3 = Conv2d(in_channels=2, out_channels=4, kernel_size=1, padding=1, stride=2) # pointwise ?

c4 = Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) # replicator
c5 = Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8) # depthwise
c6 = Conv2d(in_channels=8, out_channels=5, kernel_size=1, padding=1, stride=2) # pointwise ?

c7 = Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1) # replicator
c8 = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10) # depthwise
c9 = Conv2d(in_channels=10, out_channels=5, kernel_size=1, padding=1, stride=2) # pointwise ?

c10 = Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1) # replicator
c11 = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10) # depthwise
c12 = ConvTranspose2d(in_channels=10, out_channels=4, kernel_size=1, padding=1, stride=2) # pointwise ?

c13 = Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) # replicator
c14 = Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8) # depthwise
c15 = ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, padding=1, stride=2) # pointwise ?

c16 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) # replicator
c17 = Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6) # depthwise
c18 = ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=1, padding=1, stride=2) # pointwise ?

c19 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) # replicator
c20 = Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6) # depthwise
c21 = Conv2d(in_channels=6, out_channels=1, kernel_size=1, padding=1) # pointwise ?

from torch.nn import Sequential
from torch.nn import ReLU

model = Sequential(
    c1, ReLU(),
    c2, ReLU(),
    c3, ReLU(),
    c4, ReLU(),
    c5, ReLU(),
    c6, ReLU(),
    c7, ReLU(),
    c8, ReLU(),
    c9, ReLU(),
    c10, ReLU(),
    c11, ReLU(),
    c12, ReLU(),
    c13, ReLU(),
    c14, ReLU(),
    c15, ReLU(),
    c16, ReLU(),
    c17, ReLU(),
    c18, ReLU(),
    c19, ReLU(),
    c20, ReLU(),
    c21, ReLU(),
)
# params_depthwise = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"OutlierNet uses {params_depthwise} parameters.")
# x = torch.rand(1, 1, 72, 32)
# flops = FlopCountAnalysis(model, x)
# print(f"OutlierNet uses {flops.total():,} flops.")

class Stacked(nn.Module):
    def __init__(self):
        super(Stacked, self).__init__()
        self.s = nn.Sequential(
        nn.Conv2d(1, 4, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(4, 8, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(8, 16, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(16, 1, 3, padding=1),
        nn.Sigmoid()
        )
        
    
    def forward(self, x):
        return self.s(x)

model = Stacked()

# lin1 = Linear(3, 1)
# lin2 = Linear(10, 1)
# x = torch.rand(1, 3)
# model = torch.nn.Sequential(lin1)
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"The model uses {params} parameters.")
# flops = FlopCountAnalysis(model, x)
# print(f"The model uses {flops.total():,} flops.")

# save model to onnx and test inference time with onnxruntime
import onnx
import onnxruntime as ort
import cProfile, pstats, io
from pstats import SortKey
import time

# x = torch.rand(1, 3)
x = torch.rand(1, 1, 72, 32)
torch.onnx.export(model, x, "stacked.onnx", input_names=["input"], output_names=["output"])
onnx_model = onnx.load("stacked.onnx")
onnx.checker.check_model(onnx_model)
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# options.add_session_config_entry("session.set_denormal_as_zero", "1")
# options.execution_mode.ORT_PARALLEL
# options.add_session_config_entry("session.intra_op.allow_spinning", "0")
# options.enable_profiling = True
# options.intra_op_num_threads = 1
ort_session = ort.InferenceSession("stacked.onnx", options=options)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
x = x.numpy()
# pr = cProfile.Profile()
# pr.enable()
start = time.time()
for i in range(1000):
    ort_session.run([output_name], {input_name: x})

end = time.time()
print(f"Lin inference time: {(end - start)/ 1000 *1000} ms")

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

input_onnx_model = 'stacked.onnx'
output_onnx_model = 'stacked_fp16.onnx'

onnx_model = onnxmltools.utils.load_model(input_onnx_model)
onnx_model = convert_float_to_float16(onnx_model)
onnxmltools.utils.save_model(onnx_model, output_onnx_model)
session2 = ort.InferenceSession(output_onnx_model, options=options)
x = x.astype(np.float16)
start = time.time()
for i in range(1000):
    y = session2.run(None, {input_name: x})[0]
end = time.time()
print("Average time of inference ort_fp16: ", (end - start) / 1000 * 1000, "ms")


# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# filename = 'profile.prof'
# pr.dump_stats(filename)
# print(s.getvalue())
# t = ps.get_stats_profile().func_profiles['run'].cumtime / 10000 * 1000 #time in ms
# print(f'Average time per run: {t:.5f} ms')