"""
Machine learning module for watermelon sound classifier.
"""

from .training import (
    ModelTrainer,
    ModelConfig,
    TrainingResult,
    ModelArtifact
)

__all__ = [
    'ModelTrainer',
    'ModelConfig', 
    'TrainingResult',
    'ModelArtifact'
]