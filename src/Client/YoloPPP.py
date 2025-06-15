import cv2
import numpy as np


class YoloPPP:
    def __init__(self, mod_input_height: int, mod_input_width: int):
        self.mod_input_height = mod_input_height
        self.mod_input_width = mod_input_width

    def compute_ratio(self, input_image: np.ndarray):
        shape = input_image.shape[:2]
        new_shape = (self.mod_input_height, self.mod_input_width)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        return ratio

    def compute_unpad_shape(self, input_image: np.ndarray, ratio: float):
        shape = input_image.shape[:2]
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))

        return new_unpad

    def compute_pad(self, input_image: np.ndarray, ratio: float, unpad_shape: tuple):
        new_shape = (self.mod_input_height, self.mod_input_width)
        pad_w, pad_h = (new_shape[1] - unpad_shape[0]) / 2, (
            new_shape[0] - unpad_shape[1]
        ) / 2  # wh padding

        return pad_w, pad_h
        pass

    def preprocess(self, original_image: np.ndarray) -> np.ndarray:
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = original_image.shape[:2]  # original image shape

        ratio = self.compute_ratio(original_image)
        new_unpad = self.compute_unpad_shape(original_image, ratio)
        pad_w, pad_h = self.compute_pad(original_image, ratio, new_unpad)

        if shape[::-1] != new_unpad:  # resize
            original_image = cv2.resize(
                original_image, new_unpad, interpolation=cv2.INTER_LINEAR
            )
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        original_image = cv2.copyMakeBorder(
            original_image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        original_image = (
            np.ascontiguousarray(
                np.einsum("HWC->CHW", original_image)[::-1], dtype=np.float32
            )
            / 255.0
        )
        img_process = (
            original_image[None] if len(original_image.shape) == 3 else original_image
        )

        return img_process

    def postprocess(
        self,
        original_image: np.ndarray,
        predictions: np.ndarray,
        prototypes: np.ndarray,
        score_thr: float,
        iou_thr: float,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method for postprocessing Yolo Model Output.
        Returns boxes, masks, segments.
            @return boxes : np.ndarray of shape (N, 6) where (x1, y1, x2, y2, score, class)
        """

        ratio = self.compute_ratio(original_image)
        new_unpad = self.compute_unpad_shape(original_image, ratio)
        pad_w, pad_h = self.compute_pad(original_image, ratio, new_unpad)

        x, protos = predictions, prototypes

        # Transpose dim 1: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4 : (4 + num_classes)], axis=-1) > score_thr]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        # This is just a concatenate
        x = np.c_[
            x[..., :4],
            np.amax(x[..., 4 : (4 + num_classes)], axis=-1),
            np.argmax(x[..., 4 : (4 + num_classes)], axis=-1),
            x[..., (4 + num_classes) :],
        ]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], score_thr, iou_thr)]

        boxes = None
        segments = None
        masks = None
        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= ratio

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, original_image.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, original_image.shape[0])

            boxes = x[..., :6]

        if len(x) > 0 and protos is not None:
            # Decode masks

            # Process masks
            masks = self.__process_mask(
                protos[0], x[:, 6:], x[:, :4], original_image.shape
            )

            # Masks -> Segments(contours)
            segments = self.__masks2segments(masks)

        return boxes, masks, segments

    @staticmethod
    def __masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                0
            ]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def __crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def __process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = (
            np.matmul(masks_in, protos.reshape((c, -1)))
            .reshape((-1, mh, mw))
            .transpose(1, 2, 0)
        )  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.__scale_mask(
            masks, im0_shape
        )  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.__crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def __scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(
                im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
            )  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
                im1_shape[0] - im0_shape[0] * gain
            ) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
            round(im1_shape[1] - pad[0] + 0.1)
        )
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    # def __draw_and_visualize(self, im, bboxes, segments) -> np.ndarray:
    #     """
    #     Draw and visualize results.

    #     Args:
    #         im (np.ndarray): original image, shape [h, w, c].
    #         bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
    #         segments (List): list of segment masks.
    #         vis (bool): imshow using OpenCV.
    #         save (bool): save image annotated.

    #     Returns:
    #         None
    #     """
    #     # Draw rectangles and polygons
    #     im_canvas = im.copy()
    #     for (*box, conf, cls_), segment in zip(bboxes, segments):

    #         color = self.get_color(int(cls_))

    #         ## Draw white border
    #         cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)

    #         # draw bbox rectangle
    #         cv2.rectangle(
    #             im,
    #             (int(box[0]), int(box[1])),
    #             (int(box[2]), int(box[3])),
    #             color,
    #             1,
    #             cv2.LINE_AA,
    #         )
    #         cv2.putText(
    #             im,
    #             f"{self.classes[cls_]}: {conf:.3f}",
    #             (int(box[0]), int(box[1] - 9)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7,
    #             color,
    #             2,
    #             cv2.LINE_AA,
    #         )

    #         ## Fill with class color
    #         cv2.fillPoly(im_canvas, np.int32([segment]), color)

    #     # Overlap and mix images
    #     out_image = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

    #     return out_image
