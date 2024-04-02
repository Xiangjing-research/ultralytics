# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from pathlib import Path

from ultralytics.engine.results import Results
from ultralytics.models.yolo.multispectral import MultispectralDetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.files import increment_path


class MultispectralSegmentationPredictor(MultispectralDetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        rgb, ir = im.split(1, 0)
        return self.model(torch.cat([rgb, ir], dim=1), augment=self.args.augment, visualize=visualize)


if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO(model='v8_multispectral/train_seg/weights/best.pt', task='multispectral_seg')
    model.predict(
        ['D:\\DataSets\\PST900_RGBT_Dataset\\train\\images\\rgb\\37_bag7_rect_rgb_frame0000001732.png',
         'D:\\DataSets\\PST900_RGBT_Dataset\\train\\images\\thermal\\37_bag7_rect_thermal_frame0000001732.png'],
        save=True, name='predict')
