# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel, MultispectralDetectionModel, MultispectralSegmentationModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, },
            'multispectral': {
                'model': MultispectralDetectionModel,
                'trainer': yolo.multispectral.MultispectralDetectionTrainer,
                'validator': yolo.multispectral.MultispectralDetectionValidator,
                'predictor': yolo.multispectral.MultispectralDetectionPredictor,
            },
            'multispectral_seg': {
                'model': MultispectralSegmentationModel,
                'trainer': yolo.multispectral_seg.MultispectralSegmentationTrainer,
                'validator': yolo.multispectral_seg.MultispectralSegmentationValidator,
                'predictor': yolo.multispectral_seg.MultispectralSegmentationPredictor,
            }

        }
