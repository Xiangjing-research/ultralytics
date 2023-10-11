# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment, multispectral

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO', 'multispectral'
