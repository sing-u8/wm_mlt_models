"""
Data processing module for watermelon sound classifier.
"""

from .augmentation import (
    augment_noise, 
    AudioAugmentor, 
    BatchAugmentor,
    AugmentationResult
)

from .pipeline import (
    DataPipeline,
    AudioFile,
    DatasetSplit
)

__all__ = [
    'augment_noise', 
    'AudioAugmentor', 
    'BatchAugmentor',
    'AugmentationResult',
    'DataPipeline',
    'AudioFile',
    'DatasetSplit'
]