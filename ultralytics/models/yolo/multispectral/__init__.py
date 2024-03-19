# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import MultispectralDetectionPredictor
from .train import MultispectralDetectionTrainer
from .val import MultispectralDetectionValidator

__all__ = 'MultispectralDetectionPredictor', 'MultispectralDetectionTrainer', 'MultispectralDetectionValidator'