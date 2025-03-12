import abc

import numpy as np


class YoloPPP(abc.ABC):

    def __init__(
        self, mod_inp_height: int, mod_inp_width: int, classes: dict[int, str]
    ):
        super().__init__()

        self.mod_input_height = mod_inp_height
        self.mod_input_width = mod_inp_width

        # Load the class names from the COCO dataset
        self.classes = classes

        generator = np.random.default_rng(seed=1)
        self.color_palette = generator.uniform(0, 255, size=(len(self.classes), 3))

    @abc.abstractmethod
    def preprocess(self, input_image: np.ndarray) -> dict[str]:
        pass

    @abc.abstractmethod
    def postprocess(
        self,
        original_input,
        model_output: list[np.ndarray],
        confidence_thres: float,
        iou_thres: float,
        **kwargs
    ) -> np.ndarray:
        pass

    def get_color(self, cls_idx) -> np.ndarray:
        return self.color_palette[cls_idx]
