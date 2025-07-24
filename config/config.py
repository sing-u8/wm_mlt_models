"""
Configuration management for watermelon sound classifier.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
from pathlib import Path


@dataclass
class Config:
    """Configuration class for watermelon sound classification system."""
    
    # 데이터 경로
    data_root_dir: str = "data"
    raw_data_dir: str = "data/raw"
    noise_dir: str = "data/noise"
    processed_dir: str = "data/processed"
    model_output_dir: str = "data/models"
    
    # 오디오 처리 파라미터
    sample_rate: int = 22050
    hop_length: int = 512
    n_mfcc: int = 13
    n_chroma: int = 12
    
    # 데이터 증강 설정
    snr_levels: List[float] = field(default_factory=lambda: [-5, 0, 5, 10])
    augmentation_factor: int = 4
    
    # 모델 훈련 파라미터
    cv_folds: int = 5
    test_size: float = 0.1
    val_size: float = 0.2
    random_state: int = 42
    
    # 성능 설정
    n_jobs: int = -1  # 사용 가능한 모든 코어 사용
    batch_size: int = 32
    
    # 모델 저장 및 변환 설정
    save_pickle: bool = True
    convert_to_coreml: bool = True
    model_version: str = "1.0"
    
    # 클래스 정보
    class_names: List[str] = field(default_factory=lambda: ["watermelon_A", "watermelon_B", "watermelon_C"])
    
    def __post_init__(self):
        """초기화 후 경로들을 절대 경로로 변환."""
        base_path = Path(__file__).parent.parent
        self.data_root_dir = str(base_path / self.data_root_dir)
        self.raw_data_dir = str(base_path / self.raw_data_dir)
        self.noise_dir = str(base_path / self.noise_dir)
        self.processed_dir = str(base_path / self.processed_dir)
        self.model_output_dir = str(base_path / self.model_output_dir)
    
    def get_class_directories(self) -> Dict[str, str]:
        """클래스별 원본 데이터 디렉토리 경로 반환."""
        return {
            class_name: os.path.join(self.raw_data_dir, class_name)
            for class_name in self.class_names
        }
    
    def get_noise_subdirectories(self) -> Dict[str, str]:
        """소음 타입별 디렉토리 경로 반환."""
        return {
            "environmental": os.path.join(self.noise_dir, "environmental"),
            "mechanical": os.path.join(self.noise_dir, "mechanical"),
            "background": os.path.join(self.noise_dir, "background")
        }
    
    def get_processed_directories(self) -> Dict[str, str]:
        """처리된 데이터 디렉토리 경로 반환."""
        return {
            "augmented": os.path.join(self.processed_dir, "augmented"),
            "features": os.path.join(self.processed_dir, "features"),
            "splits": os.path.join(self.processed_dir, "splits")
        }
    
    def get_model_directories(self) -> Dict[str, str]:  
        """모델 저장 디렉토리 경로 반환."""
        return {
            "artifacts": os.path.join(self.model_output_dir, "artifacts"),
            "pickle": os.path.join(self.model_output_dir, "pickle"),
            "coreml": os.path.join(self.model_output_dir, "coreml")
        }


# SVM 하이퍼파라미터 그리드
SVM_PARAMS = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

# Random Forest 하이퍼파라미터 그리드
RF_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 기본 설정 인스턴스
DEFAULT_CONFIG = Config()