import csv

import keras
import numpy as np
import onnx
import onnx2tf
import onnx_tool
import onnxruntime
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


def profile():
    # yolo = YOLO()
    # yolo.export(format="onnx", dynamic=True)
    # onnxModel = onnx.load_model("./yolo11n.onnx")
    # graph = onnxModel.graph
    # nodes = list(graph.node)
    # nodeNames = set([node.name for node in nodes])

    # m = onnx_tool.Model("./yolo11n.onnx")
    # m.graph.shape_infer(
    #     {"images": np.zeros((1, 3, 224, 224))}
    # )  # update tensor shapes with new input tensor
    # m.graph.profile()
    # m.graph.print_node_map()

    sess = onnxruntime.InferenceSession("./yolo11n.onnx")
    outs = sess.run(
        None, input_feed={"images": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    model: torch.Module = torch.load("yolo11n.pt")["model"]
    model(np.zeros((1, 3, 224, 224)))
    # with torch.no_grad():  # Disable gradient computation
    #     model(np.zeros((1, 3, 224, 224)))


def main():
    # Load a mode
    model: YOLO = YOLO(
        "yolo11n.pt"
    )  # load a pretrained model (recommended for training)
    model.export(format="onnx")
    kerasModel = onnx2tf.convert("./yolo11n.onnx", output_keras_v3=True)
    return
    onnxModel: onnx.ModelProto = onnx.load_model("./yolo11n.onnx")
    sess = onnxruntime.InferenceSession("./yolo11n.onnx")
    graph = onnxModel.graph

    nodes = list(graph.node)
    inputsDict = {}
    outputsDict = {}
    for node in nodes:
        for inp in list(node.input):
            inputsDict.setdefault(inp, [])
            inputsDict[inp].append(node)
        for out in list(node.output):
            outputsDict.setdefault(out, [])
            outputsDict[out].append(node)

    totModels = 0
    for i in range(0, len(nodes), 50):
        print(f"Exporting SubModel {i // 50}")
        subNodes = nodes[i : min(i + 50, len(nodes))]
        subModOuts = set()
        subModInps = set()
        for subNode in subNodes:
            for outName in list(subNode.output):
                nextNodes = inputsDict.get(outName, None)
                if nextNodes is None:
                    if outName in [out.name for out in graph.output]:
                        subModOuts.add(outName)
                else:
                    for nextNode in inputsDict[outName]:
                        if nextNode not in subNodes:
                            subModOuts.add(outName)

            for inpName in list(subNode.input):
                prevNodes = outputsDict.get(inpName, None)
                if prevNodes is None:
                    ## Graph Input tensor
                    if inpName in [inp.name for inp in graph.input]:
                        subModInps.add(inpName)
                else:
                    for prevNode in prevNodes:
                        if prevNode not in subNodes:
                            subModInps.add(inpName)
        print(subModInps)
        print(subModOuts)
        onnx.utils.extract_model(
            "./yolo11n.onnx",
            f"./sub_model_{i // 50}.onnx",
            input_names=subModInps,
            output_names=subModOuts,
            check_model=True,
        )
        totModels += 1

    partialResDict = {"images": np.ones(shape=(1, 3, 640, 640), dtype=np.float32)}
    for i in range(0, totModels):
        sess = onnxruntime.InferenceSession(f"./sub_model_{i}.onnx")
        inpNames = [inp.name for inp in sess.get_inputs()]
        inpDict = {inpName: partialResDict[inpName] for inpName in inpNames}

        outList = sess.run(None, input_feed=inpDict)

        outNames = [out.name for out in sess.get_outputs()]
        outDict = {outNames[i]: outList[i] for i in range(0, len(outNames))}
        partialResDict.update(outDict)

    print(outDict)

    sess = onnxruntime.InferenceSession("./yolo11n.onnx")
    totalOutList = sess.run(
        None,
        {"images": np.ones(shape=(1, 3, 640, 640), dtype=np.float32)},
    )
    totalOutDict = {
        out.name: totalOutList[idx] for idx, out in enumerate(sess.get_outputs())
    }
    print(totalOutDict)


def main_1():
    # Load a model
    # model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    # results = model("bus.jpg")
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     # result.show()  # display to screen
    #     result.save(filename="result.jpg")  # save to disk
    model = YOLO("yolo11n-seg.pt")
    model.export(format="onnx")


if __name__ == "__main__":
    # profile()
    main()
