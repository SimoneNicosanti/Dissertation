import keras
import ModelParse

MAX_LAYERS_PER_SERVICE = 500


class ModelManager:

    def __init__(self, model: keras.Model):
        self.model = model
        self.allOps, self.opsInfoDict, self.prevOpsDict, self.nextOpsDict = (
            ModelParse.modelParse(model)
        )
        validLayersName = [layer.name for layer in model.layers]
        callables = ModelParse.buildCallables(
            self.allOps, self.opsInfoDict, model, validLayersName
        )

        ModelParse.saveCallables(callables)

    def getSubModelInfo(self, idx: int):
        start = idx * MAX_LAYERS_PER_SERVICE
        end = min(len(self.allOps), start + MAX_LAYERS_PER_SERVICE)
        opsSubList = self.allOps[start:end]
        subPrevOps = {op: self.prevOpsDict[op] for op in opsSubList}
        subNextOps = {op: self.nextOpsDict[op] for op in opsSubList}

        return opsSubList, subPrevOps, subNextOps
