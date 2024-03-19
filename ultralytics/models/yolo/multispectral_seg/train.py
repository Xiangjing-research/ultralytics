# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_multimodal_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import MultispectralSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.models.yolo.multispectral import MultispectralDetectionTrainer


class MultispectralSegmentationTrainer(MultispectralDetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        # overrides['task'] = 'segment'
        overrides['task'] = 'multispectral_seg'
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = MultispectralSegmentationModel(cfg, ch=3, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
        return yolo.multispectral_seg.MultispectralSegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                                                         args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        x = torch.split(batch['img'], 3, dim=1)
        rgb = x[0]
        ir = x[1]
        plot_images(images=rgb,
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    masks=batch['masks'],
                    fname=self.save_dir / f'train_rgb_batch{ni}.jpg',
                    on_plot=self.on_plot)
        for i, im_file in enumerate(batch['im_file']):
            batch['im_file'][i] = im_file.replace('visible', 'infrared')
        plot_images(images=ir,
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    masks=batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_ir_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png


if __name__ == '__main__':
    from ultralytics import YOLO

    # model = YOLO(model='../../../cfg/models/v8/yolov8l-C2f_FasterNet-DFMDA-seg.yaml', task='multispectral_seg')
    # model.train(data='../../../cfg/datasets/PST900.yaml', epochs=300, batch=4)
    # data = build_multimodal_dataset(cfg=DEFAULT_CFG, rgb_path='D:\\DataSets\\PST900_RGBT_Dataset\\train\\train-rgb.txt', ir_path='D:\\DataSets\\PST900_RGBT_Dataset\\train\\train-thermal.txt', batch=8, data=check_det_dataset(get_cfg().data))

    from ultralytics.models.yolo.multispectral_seg import MultispectralSegmentationTrainer

    args = dict(task='multispectral_seg', mode='train', model='../../../cfg/models/v8/yolov8l-C2f_FasterNet-DFMDA-seg.yaml',
                data=' ../../../cfg/datasets/PST900.yaml', epochs=300, batch=4, project='v8_multispectral',
                name='train_seg')
    trainer = MultispectralSegmentationTrainer(overrides=args)
    trainer.train()