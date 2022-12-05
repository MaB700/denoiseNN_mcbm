import onnx
import onnxruntime
import onnxruntime.quantization as oq

quantized_onnx_model = oq.quantize_static('./mixed.onnx', './mixed_quantized.onnx', weight_type=oq.QuantType.QUInt8)
