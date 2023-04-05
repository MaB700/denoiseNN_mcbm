import onnxruntime as ort
import numpy as np
import time
sess = ort.InferenceSession("efficientformer_l1.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.random.random((1, 3, 224, 224)).astype(np.float32)

start = time.time()
for i in range(1000):
    pred_ort = sess.run([label_name], {input_name: x})[0]
print(pred_ort)
print("Time: ", (time.time() - start) / 1000.0 * 1000.0, " ms")