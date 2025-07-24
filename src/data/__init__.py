"""
Data processing module for watermelon sound classifier.
"""

from .augmentation import (
    augment_noise, 
    AudioAugmentor, 
    BatchAugmentor,
    AugmentationResult
)

__all__ = [
    'augment_noise', 
    'AudioAugmentor', 
    'BatchAugmentor',
    'AugmentationResult'
]