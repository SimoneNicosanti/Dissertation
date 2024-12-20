import keras
from Manipulation import PathGenerator, Utils


class OperationWrapper:
    def __init__(
        self, operation: keras.Operation, model: keras.Model, operationPath: str
    ):
        self.op: keras.Operation = operation
        self.opModel: keras.Model = model  ## Model the op belongs to
        self.opPath: str = operationPath  ## Path of the operation inside the model

    def isKerasModel(self) -> bool:
        return isinstance(self.op, keras.Model) or isinstance(self.op, keras.Sequential)

    def getName(self) -> str:
        return self.opPath

    def getOp(self) -> keras.Operation:
        return self.op

    def belongsToModel(self, model: keras.Model) -> bool:
        return model.name == self.opModel.name

    def isOutputOp(self) -> bool:
        if isinstance(self.opModel, keras.Sequential):
            return self.op.name == self.opModel.layers[-1].name
        elif isinstance(self.opModel, keras.Model):
            return self.op.name in self.opModel.output_names

    def isInputOp(self) -> bool:
        return isinstance(self.op, keras.layers.InputLayer)

    def getNextOpsPaths(self) -> list[keras.Operation]:
        basePath = PathGenerator.getBasePath(self.opPath)
        return [
            PathGenerator.generatePath(basePath, node.operation.name)
            for node in self.op._outbound_nodes
        ]

    def getSubModelInputNames(self):
        if not self.isKerasModel():
            raise ValueError("This Operation is Not a Keras Model")

        ## Getting inputs as received by sub model
        subModInputs = self.op.inputs
        subModelInputLayers = [
            inp._keras_history.operation.name for inp in subModInputs
        ]

        subModelInputPaths = [
            PathGenerator.generatePath(self.opPath, name)
            for name in subModelInputLayers
        ]

        return subModelInputPaths

    def getPrevLayersNames(self):
        layerInputs = self.op._inbound_nodes[0].arguments._flat_arguments
        basePath = PathGenerator.getBasePath(self.opPath)
        layerInputsPrevNames = [
            inp._keras_history.operation.name
            for inp in layerInputs
            if isinstance(inp, keras.KerasTensor)
        ]

        layerInputsPrevPaths = [
            PathGenerator.generatePath(basePath, name) for name in layerInputsPrevNames
        ]
        return layerInputsPrevPaths

    def getOpOutput(self):
        return self.op.output

    def getArguments(self) -> tuple[list, dict]:
        if self.isKerasModel():
            ## It is a sub model
            ## We change the sub model with an Identity Layer
            ## returning the same output as the sub model itself
            return [self.op.outputs], {}
        elif self.isInputOp():
            ## It is input layer of sub model
            ## We chnage it with an Identity layer returning
            ## the same output as the sub model
            subModInputs: list[str] = Utils.getInputLayersNames(self.opModel)
            inputIdx: int = subModInputs.index(self.op.name)

            ## TODO >> Check this if is enough general
            for argElem in self.opModel._inbound_nodes[0].arguments.args:
                if isinstance(argElem, list):
                    return [argElem[inputIdx]], {}
                else:
                    return [argElem], {}
        else:
            ## Simple operation
            ## Return its args
            return (
                self.op._inbound_nodes[0].arguments.args,
                self.op._inbound_nodes[0].arguments.kwargs,
            )

    def getCallable(self) -> keras.Operation:
        ## In order to keep the model original struct, we change both
        ## input layers and layers representing sub models with IdentityLayers
        newOperation: keras.Operation = None
        if self.isKerasModel() or self.isInputOp():
            newOperation = keras.layers.Identity(name=self.getName())
        else:
            newOperation = self.op
        return newOperation
