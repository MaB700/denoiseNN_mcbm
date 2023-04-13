import numpy
import time
import torch
import onnxruntime as ort

def test_latency(model, data, num_samples=1000):
    model.eval()
    file_name = "model.onnx"
    dynamic_axes = {"nodes": {0: "num_nodes"}, 
                "edge_index": {1: "num_edges"}, 
                "edge_attr": {0: "num_edges"},
                "output": {0: "num_nodes"}}

    input_data = (data[0].x, data[0].edge_index, data[0].edge_attr)
    torch.onnx.export(model, input_data, file_name, input_names=["nodes", "edge_index", "edge_attr"], opset_version=16,
                  output_names=["output"], export_params=True, dynamic_axes=dynamic_axes)
    print("exported model to", file_name)
    expected = model(*input_data)
    
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_profiling = False
    options.execution_mode.ORT_SEQUENTIAL
    session = ort.InferenceSession(file_name, sess_options=options)
    out = session.run(None, {"nodes": input_data[0].numpy(), "edge_index": input_data[1].numpy(), "edge_attr": input_data[2].numpy()})[0]

    print("allclose:", torch.allclose(torch.from_numpy(out), expected))

    x_list = []
    edge_index_list = []
    edge_attr_list = []
    for i in range(num_samples):
        x_list.append(data[i].x.detach().numpy())
        edge_index_list.append(data[i].edge_index.detach().numpy())
        edge_attr_list.append(data[i].edge_attr.detach().numpy())

    start = time.time()
    for i in range(num_samples):
        session.run(None, {"nodes": x_list[i], "edge_index": edge_index_list[i], "edge_attr": edge_attr_list[i]})[0]
    end = time.time()
    delta_t = (end - start) *1000 / num_samples
    print("Average time of inference x: ", delta_t, "ms")
    return delta_t