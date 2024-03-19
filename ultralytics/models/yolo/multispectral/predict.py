# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import deepcopy
from pathlib import Path

import cv2
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops, LOGGER, colorstr
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import smart_inference_mode


class MultispectralDetectionPredictor(DetectionPredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        preds = [preds[0], deepcopy(preds[0])]
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        rgb, ir = im.split(1, 0)
        return self.model(torch.cat([rgb, ir], dim=1), augment=self.args.augment, visualize=visualize)



if __name__ == '__main__':
    from ultralytics import YOLO

    # model = YOLO(model='v8_multispectral/train-C2f_FasterNet-DFMDA-LLVIP4/weights/best.pt', task='multispectral')
    # model.predict(
    #     ['D:\\DataSets\\LLVIP\\images\\visible\\train\\020344.jpg', 'D:\\DataSets\\LLVIP\\images\\infrared\\train\\020344.jpg'],
    #     save=True, name='predict')