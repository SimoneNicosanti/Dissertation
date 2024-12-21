import keras


def getModelOperations(model):
    if isinstance(model, keras.Sequential):
        hiddenInputLayer = model.layers[0].input._keras_history.operation
        allLayers = [hiddenInputLayer]
        allLayers.extend(model.layers)
        return allLayers
    elif isinstance(model, keras.Model):
        return model.operations
    else:
        raise ValueError("The Model Is Not A Valid Instance")


def getModelOutputLayersNames(model):
    if isinstance(model, keras.Sequential):
        return [model.layers[-1].name]
    elif isinstance(model, keras.Model):
        return model.output_names
    else:
        raise ValueError("The Model Is Not A Valid Instance")


def getModelInputLayersNames(model: keras.Model | keras.Sequential) -> list[str]:
    if isinstance(model, keras.Sequential):
        hiddenLayer = model.layers[0].input._keras_history.operation
        return [hiddenLayer.name]

    inputLayerNames = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            inputLayerNames.append(layer.name)
    return inputLayerNames


def convertToList(anyValue: list[keras.KerasTensor] | dict[str, keras.KerasTensor]):
    if isinstance(anyValue, list):
        sortedList = sorted(anyValue, key=lambda x: x.name)
        return sortedList
    elif isinstance(anyValue, dict):
        valueList: list[keras.KerasTensor] = list(anyValue.values())
        sortedList = sorted(valueList, key=lambda x: x.name)
        return sortedList
    else:
        return [anyValue]
