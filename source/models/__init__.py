"""Model artifacts and their model-specific loading/inference code.

    detector/   YOLO11n person detector (Ultralytics .pt)
    reid/       PS-1 ResNet-18 ReID checkpoint (PyTorch) + shared preprocessing
    onnx/       ReID exported to ONNX (ONNX Runtime backends; TensorRT build source)
    tensorrt/   TensorRT engines built from the ONNX model + engine builder

``model_registry.py`` creates and caches them (detector + auto-selected ReID backend).
"""
