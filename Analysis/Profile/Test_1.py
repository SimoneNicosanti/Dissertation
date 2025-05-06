import numpy as np
import onnx_tool
import onnxruntime


def main():
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.optimized_model_filepath = "yolo11n-seg-optimized.onnx"
    sess = onnxruntime.InferenceSession(
        "yolo11n-seg_quant.onnx", sess_options=sess_options
    )

    sess.run(None, input_feed={"images": np.zeros((1, 3, 640, 640), dtype=np.float32)})

    sess.end_profiling()

    pass


if __name__ == "__main__":
    main()
