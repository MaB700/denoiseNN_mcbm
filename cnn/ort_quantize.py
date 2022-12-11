import numpy as np
import uproot

import onnx
import onnxruntime
import onnxruntime.quantization as oq
from onnxruntime.quantization.calibrate import CalibrationDataReader

class CalibrationDataProvider(CalibrationDataReader):
    def __init__(self):
        super(CalibrationDataProvider, self).__init__()
        self.counter = 0
        self.x = None
        with uproot.open("../data.root") as file:
            self.x = np.reshape(np.array(file["train"]["time"].array(entry_stop=1000)), (-1, 72, 32, 1))

    def get_next(self):
        if self.counter > 1000 - 1:
            return None
        else:
            out = {'input_1': self.x[self.counter][np.newaxis]}
            self.counter += 1
            return out


cdp = CalibrationDataProvider()
quantized_onnx_model = oq.quantize_static('./mixed.onnx', './mixed_quantized.onnx', weight_type=oq.QuantType.QInt8, calibration_data_reader=cdp, per_channel=True, reduce_range=True)
