import abc

from Model.Model_IO import ModelInput, ModelOutput


class ModelRunner(abc.ABC):
    def __init__(self, sub_model_path: str):
        self.sub_model_path = sub_model_path

        self.input_names = []
        self.output_names = []

    @abc.abstractmethod
    def run(self, model_input: ModelInput) -> ModelOutput:
        return

    def get_input_names(self) -> list[str]:
        return self.input_names

    def get_output_names(self) -> list[str]:
        return self.output_names
