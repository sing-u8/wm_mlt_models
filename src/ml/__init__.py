"""
Machine learning module for watermelon sound classifier.
"""

from .training import (
    ModelTrainer,
    ModelConfig,
    TrainingResult,
    ModelArtifact
)

from .evaluation import (
    ModelEvaluator,
    ClassificationMetrics,
    ModelComparison,
    EvaluationReport
)

__all__ = [
    'ModelTrainer',
    'ModelConfig', 
    'TrainingResult',
    'ModelArtifact',
    'ModelEvaluator',
    'ClassificationMetrics', 
    'ModelComparison',
    'EvaluationReport'
]