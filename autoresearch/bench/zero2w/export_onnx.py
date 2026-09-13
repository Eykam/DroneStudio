#!/usr/bin/env python3
"""Box1-side export: VisNet checkpoint -> ONNX for the Zero 2 W bench.
Run with the vision venv: /workspace/venv-vision/bin/python export_onnx.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from vis_train import VisNet

CKPT = os.environ.get("VIS_CKPT", "/workspace/vision_model/visnet_v11_best.pt")
OUT = os.path.join(os.path.dirname(__file__), "visnet_v11_128x96.onnx")

net = VisNet()
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
net.eval()
dummy = torch.zeros(1, 3, 96, 128)
torch.onnx.export(
    net, dummy, OUT, opset_version=17,
    input_names=["rgb"], output_names=["depth", "seg"],
    dynamic_axes=None,
)
print("exported", OUT, os.path.getsize(OUT), "bytes")

# verify: onnxruntime vs torch on the same input
import numpy as np, onnxruntime as ort
x = np.random.default_rng(0).random((1, 3, 96, 128), dtype=np.float32)
with torch.no_grad():
    td, ts = net(torch.from_numpy(x))
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])
od, os_ = sess.run(None, {"rgb": x})
print("verify depth max|dt-onnx| =", float(np.abs(td.numpy() - od).max()))
print("verify seg max|dt-onnx| =", float(np.abs(ts.numpy() - os_).max()))
