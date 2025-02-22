import numpy as np


class ModelInput:
    def __init__(self):
        self.input_dict = {}

    def get_all_inputs(self) -> dict[str, np.ndarray]:
        return self.input_dict

    def get_input_by_name(self, input_name: str) -> np.ndarray:
        return self.input_dict[input_name]

    def add_input(self, input_name: str, input_value: np.ndarray):
        self.input_dict[input_name] = input_value

    def extend_input(self, other_model_input) -> None:
        self.input_dict.update(other_model_input.get_all_inputs())

    def get_all_input_names(self) -> list[str]:
        return list(self.input_dict.keys())


class ModelOutput:
    def __init__(self, output_dict: dict[str, np.ndarray]):
        self.output_dict = output_dict

    def get_all_outputs(self) -> dict[str, np.ndarray]:
        return self.output_dict

    def get_output_by_name(self, output_name: str) -> np.ndarray:
        return self.output_dict[output_name]
