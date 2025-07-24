"""
데이터 파이프라인 관리 모듈

이 모듈은 적절한 순서를 통해 데이터 흐름을 관리하고 
데이터 누출을 방지하는 포괄적인 파이프라인을 구현합니다.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from collections import defaultdict

from ..utils.logger import LoggerMixin
from ..audio.feature_extraction import extract_features, FeatureVector
from .augmentation import BatchAugmentor, AugmentationResult
from config import DEFAULT_CONFIG


@dataclass
class AudioFile:
    """
    오디오 파일 메타데이터를 관리하는 데이터 클래스.
    """
    file_path: str
    class_name: str
    split: str  # 'train', 'validation', 'test'
    is_augmented: bool = False
    original_file: Optional[str] = None  # 증강된 경우 원본 파일 경로
    noise_type: Optional[str] = None     # 사용된 소음 타입
    snr_level: Optional[float] = None    # 사용된 SNR 레벨
    file_size: Optional[int] = None      # 파일 크기 (bytes)
    duration: Optional[float] = None     # 오디오 길이 (seconds)
    
    def __post_init__(self):
        """파일 메타데이터 자동 설정"""
        if os.path.exists(self.file_path):
            self.file_size = os.path.getsize(self.file_path)
        
        # 증강 파일에서 정보 추출
        if self.is_augmented and "_noise_" in self.file_path:
            filename = Path(self.file_path).stem
            parts = filename.split("_")
            
            # SNR 정보 추출 (예: snr+5dB)
            for part in parts:
                if part.startswith("snr"):
                    try:
                        snr_str = part.replace("snr", "").replace("dB", "")
                        self.snr_level = float(snr_str)
                    except ValueError:
                        pass


@dataclass
class DatasetSplit:
    """
    데이터셋 분할 정보를 담는 데이터 클래스.
    """
    train_files: Dict[str, List[AudioFile]]
    validation_files: Dict[str, List[AudioFile]]
    test_files: Dict[str, List[AudioFile]]
    noise_files: List[str]
    total_train: int = 0
    total_validation: int = 0
    total_test: int = 0
    
    def __post_init__(self):
        """전체 파일 수 계산"""
        self.total_train = sum(len(files) for files in self.train_files.values())
        self.total_validation = sum(len(files) for files in self.validation_files.values())
        self.total_test = sum(len(files) for files in self.test_files.values())


class DataPipeline(LoggerMixin):
    """
    데이터 파이프라인 관리 클래스.
    
    design.md에 명시된 인터페이스를 구현하고
    데이터 누출 방지를 보장합니다.
    """
    
    def __init__(self, config=None):
        """
        데이터 디렉토리로 파이프라인을 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        
        # 경로 설정
        self.raw_data_dir = self.config.raw_data_dir
        self.noise_dir = self.config.noise_dir
        self.processed_dir = self.config.processed_dir
        
        # 컴포넌트 초기화
        self.batch_augmentor = BatchAugmentor(config)
        
        # 상태 관리
        self._dataset_split = None
        self._augmentation_results = {}
        
        self.logger.info(f"DataPipeline 초기화됨")
        self.logger.info(f"  원본 데이터: {self.raw_data_dir}")
        self.logger.info(f"  소음 데이터: {self.noise_dir}")
        self.logger.info(f"  처리된 데이터: {self.processed_dir}")
    
    def _load_class_files(self, data_dir: str, split_name: str) -> Dict[str, List[AudioFile]]:
        """
        지정된 디렉토리에서 클래스별 오디오 파일을 로드합니다.
        
        Parameters:
        -----------
        data_dir : str
            데이터 디렉토리 경로
        split_name : str
            분할 이름 ('train', 'validation', 'test')
            
        Returns:
        --------
        Dict[str, List[AudioFile]]
            클래스별 오디오 파일 딕셔너리
        """
        class_files = {}
        
        if not os.path.exists(data_dir):
            self.logger.warning(f"디렉토리가 존재하지 않음: {data_dir}")
            return class_files
        
        for class_name in self.config.class_names:
            class_dir = os.path.join(data_dir, class_name)
            audio_files = []
            
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith('.wav'):
                        file_path = os.path.join(class_dir, file_name)
                        
                        audio_file = AudioFile(
                            file_path=file_path,
                            class_name=class_name,
                            split=split_name,
                            is_augmented=False
                        )
                        audio_files.append(audio_file)
            
            class_files[class_name] = audio_files
            self.logger.debug(f"{split_name} {class_name}: {len(audio_files)}개 파일")
        
        return class_files
    
    def load_train_data(self) -> Dict[str, List[AudioFile]]:
        """
        data/raw/train/ 디렉토리에서 클래스별 오디오 파일을 로드합니다.
        
        Returns:
        --------
        Dict[str, List[AudioFile]]
            클래스별 훈련 오디오 파일 딕셔너리
        """
        train_dir = os.path.join(self.raw_data_dir, "train")
        train_files = self._load_class_files(train_dir, "train")
        
        total_files = sum(len(files) for files in train_files.values())
        self.logger.info(f"훈련 데이터 로드 완료: {total_files}개 파일")
        
        return train_files
    
    def load_validation_data(self) -> Dict[str, List[AudioFile]]:
        """
        data/raw/validation/ 디렉토리에서 클래스별 오디오 파일을 로드합니다.
        
        Returns:
        --------
        Dict[str, List[AudioFile]]
            클래스별 검증 오디오 파일 딕셔너리
        """
        validation_dir = os.path.join(self.raw_data_dir, "validation")
        validation_files = self._load_class_files(validation_dir, "validation")
        
        total_files = sum(len(files) for files in validation_files.values())
        self.logger.info(f"검증 데이터 로드 완료: {total_files}개 파일")
        
        return validation_files
    
    def load_test_data(self) -> Dict[str, List[AudioFile]]:
        """
        data/raw/test/ 디렉토리에서 클래스별 오디오 파일을 로드합니다.
        
        Returns:
        --------
        Dict[str, List[AudioFile]]
            클래스별 테스트 오디오 파일 딕셔너리
        """
        test_dir = os.path.join(self.raw_data_dir, "test")
        test_files = self._load_class_files(test_dir, "test")
        
        total_files = sum(len(files) for files in test_files.values())
        self.logger.info(f"테스트 데이터 로드 완료: {total_files}개 파일")
        
        return test_files
    
    def load_noise_files(self) -> List[str]:
        """
        data/noise/ 디렉토리에서 사용 가능한 소음 파일을 재귀적으로 로드합니다.
        
        Returns:
        --------
        List[str]
            소음 파일 경로 목록
        """
        noise_files = self.config.get_all_noise_files()
        
        if noise_files:
            self.logger.info(f"소음 파일 로드 완료: {len(noise_files)}개 파일")
            
            # 소음 타입별 통계
            noise_stats = defaultdict(int)
            for noise_file in noise_files:
                noise_type = self.batch_augmentor._extract_noise_type(noise_file)
                noise_stats[noise_type] += 1
            
            for noise_type, count in noise_stats.items():
                self.logger.info(f"  {noise_type}: {count}개 파일")
        else:
            self.logger.warning("사용 가능한 소음 파일이 없습니다.")
        
        return noise_files
    
    def load_all_data(self) -> DatasetSplit:
        """
        모든 데이터 분할을 로드하고 DatasetSplit 객체를 반환합니다.
        
        Returns:
        --------
        DatasetSplit
            전체 데이터셋 분할 정보
        """
        self.logger.info("=== 전체 데이터 로딩 시작 ===")
        
        # 각 분할 데이터 로드
        train_files = self.load_train_data()
        validation_files = self.load_validation_data()
        test_files = self.load_test_data()
        noise_files = self.load_noise_files()
        
        # DatasetSplit 생성
        dataset_split = DatasetSplit(
            train_files=train_files,
            validation_files=validation_files,
            test_files=test_files,
            noise_files=noise_files
        )
        
        self._dataset_split = dataset_split
        
        self.logger.info("=== 전체 데이터 로딩 완료 ===")
        self.logger.info(f"  훈련: {dataset_split.total_train}개 파일")
        self.logger.info(f"  검증: {dataset_split.total_validation}개 파일")
        self.logger.info(f"  테스트: {dataset_split.total_test}개 파일")
        self.logger.info(f"  소음: {len(dataset_split.noise_files)}개 파일")
        
        return dataset_split
    
    def augment_training_data(self, noise_files: List[str] = None, 
                            force_augmentation: bool = False) -> Dict[str, List[AudioFile]]:
        """
        훈련 세트에만 증강을 적용합니다. 
        소음 파일이 없으면 원본 데이터만 사용합니다.
        
        Parameters:
        -----------
        noise_files : List[str], optional
            사용할 소음 파일 목록. None이면 자동으로 로드합니다.
        force_augmentation : bool
            소음 파일이 부족해도 강제로 증강을 수행할지 여부
            
        Returns:
        --------
        Dict[str, List[AudioFile]]
            증강된(또는 원본) 훈련 데이터
        """
        self.logger.info("=== 훈련 데이터 증강 시작 ===")
        
        # 훈련 데이터가 로드되지 않았으면 로드
        if self._dataset_split is None:
            self.load_all_data()
        
        train_files = self._dataset_split.train_files
        
        # 소음 파일 준비
        if noise_files is None:
            noise_files = self._dataset_split.noise_files
        
        # 소음 파일 검증
        if not noise_files:
            self.logger.warning("소음 파일이 없어 원본 훈련 데이터만 사용합니다.")
            return train_files
        
        if len(noise_files) < self.config.min_noise_files and not force_augmentation:
            self.logger.warning(f"소음 파일 부족 ({len(noise_files)}개 < {self.config.min_noise_files}개). "
                              f"원본 훈련 데이터만 사용합니다.")
            return train_files
        
        # 증강 출력 디렉토리 준비
        augmented_base_dir = os.path.join(self.processed_dir, "augmented")
        os.makedirs(augmented_base_dir, exist_ok=True)
        
        # 클래스별 증강 수행
        augmented_train_files = {}
        self._augmentation_results = {}
        
        for class_name, class_files in train_files.items():
            if not class_files:
                self.logger.warning(f"클래스 {class_name}에 훈련 파일이 없습니다.")
                augmented_train_files[class_name] = []
                continue
            
            self.logger.info(f"클래스 {class_name} 증강 시작 ({len(class_files)}개 원본 파일)")
            
            # 클래스 디렉토리 경로
            class_dir = os.path.join(self.raw_data_dir, "train", class_name)
            
            # 배치 증강 수행
            try:
                augmentation_result = self.batch_augmentor.augment_class_directory(
                    class_dir, class_name, noise_files, augmented_base_dir
                )
                
                self._augmentation_results[class_name] = augmentation_result
                
                # 증강된 파일들을 AudioFile 객체로 변환
                augmented_audio_files = []
                
                # 원본 파일들 추가
                augmented_audio_files.extend(class_files)
                
                # 증강된 파일들 추가
                for augmented_path in augmentation_result.augmented_files:
                    augmented_file = AudioFile(
                        file_path=augmented_path,
                        class_name=class_name,
                        split="train",
                        is_augmented=True,
                        original_file=class_dir  # 원본 클래스 디렉토리
                    )
                    augmented_audio_files.append(augmented_file)
                
                augmented_train_files[class_name] = augmented_audio_files
                
                self.logger.info(f"클래스 {class_name} 증강 완료: "
                               f"{len(class_files)}개 원본 + {augmentation_result.total_created}개 증강 "
                               f"= {len(augmented_audio_files)}개 총 파일")
                
            except Exception as e:
                self.logger.error(f"클래스 {class_name} 증강 실패: {e}")
                # 실패 시 원본 파일만 사용
                augmented_train_files[class_name] = class_files
        
        # 전체 통계
        total_original = sum(len(files) for files in train_files.values())
        total_augmented = sum(len(files) for files in augmented_train_files.values())
        
        self.logger.info("=== 훈련 데이터 증강 완료 ===")
        self.logger.info(f"  원본: {total_original}개 파일")
        self.logger.info(f"  총 파일: {total_augmented}개 파일")
        self.logger.info(f"  증강 비율: {(total_augmented / total_original):.1f}x")
        
        return augmented_train_files
    
    def validate_data_integrity(self) -> bool:
        """
        데이터 무결성을 검증하고 데이터 누출이 없는지 확인합니다.
        
        Returns:
        --------
        bool
            데이터 무결성이 유지되면 True
        """
        self.logger.info("=== 데이터 무결성 검증 시작 ===")
        
        if self._dataset_split is None:
            self.logger.error("데이터가 로드되지 않았습니다.")
            return False
        
        integrity_issues = []
        
        # 1. 각 분할에 모든 클래스가 있는지 확인
        for split_name, split_files in [
            ("train", self._dataset_split.train_files),
            ("validation", self._dataset_split.validation_files),
            ("test", self._dataset_split.test_files)
        ]:
            missing_classes = []
            for class_name in self.config.class_names:
                if class_name not in split_files or not split_files[class_name]:
                    missing_classes.append(class_name)
            
            if missing_classes:
                integrity_issues.append(f"{split_name} 분할에서 누락된 클래스: {missing_classes}")
        
        # 2. 파일 존재 확인
        missing_files = []
        for split_name, split_files in [
            ("train", self._dataset_split.train_files),
            ("validation", self._dataset_split.validation_files),
            ("test", self._dataset_split.test_files)
        ]:
            for class_name, files in split_files.items():
                for audio_file in files:
                    if not os.path.exists(audio_file.file_path):
                        missing_files.append(f"{split_name}/{class_name}/{os.path.basename(audio_file.file_path)}")
        
        if missing_files:
            integrity_issues.append(f"존재하지 않는 파일들: {missing_files[:5]}{'... 등' if len(missing_files) > 5 else ''}")
        
        # 3. 데이터 누출 검사: 같은 파일이 여러 분할에 있는지 확인
        all_file_paths = {}
        for split_name, split_files in [
            ("train", self._dataset_split.train_files),
            ("validation", self._dataset_split.validation_files),
            ("test", self._dataset_split.test_files)
        ]:
            for class_name, files in split_files.items():
                for audio_file in files:
                    if not audio_file.is_augmented:  # 원본 파일만 검사
                        file_key = os.path.basename(audio_file.file_path)
                        if file_key in all_file_paths:
                            integrity_issues.append(f"데이터 누출 감지: {file_key}가 {all_file_paths[file_key]}와 {split_name}에 동시 존재")
                        else:
                            all_file_paths[file_key] = split_name
        
        # 4. 증강 데이터 검증: 증강이 훈련 세트에만 적용되었는지 확인
        for split_name, split_files in [
            ("validation", self._dataset_split.validation_files),
            ("test", self._dataset_split.test_files)
        ]:
            for class_name, files in split_files.items():
                augmented_in_split = [f for f in files if f.is_augmented]
                if augmented_in_split:
                    integrity_issues.append(f"{split_name} 분할에 증강 데이터 발견: {len(augmented_in_split)}개 파일")
        
        # 결과 보고
        if integrity_issues:
            self.logger.error("데이터 무결성 문제 발견:")
            for issue in integrity_issues:
                self.logger.error(f"  - {issue}")
            return False
        else:
            self.logger.info("✅ 데이터 무결성 검증 통과")
            self.logger.info("  - 모든 클래스가 각 분할에 존재")
            self.logger.info("  - 모든 파일이 존재함")
            self.logger.info("  - 데이터 누출 없음")
            self.logger.info("  - 증강이 훈련 세트에만 적용됨")
            return True
    
    def extract_all_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                         np.ndarray, np.ndarray, np.ndarray]:
        """
        모든 데이터셋에서 특징을 추출합니다.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.logger.info("=== 전체 특징 추출 시작 ===")
        
        if self._dataset_split is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_all_data()를 먼저 호출하세요.")
        
        # 클래스 라벨 매핑
        class_to_label = {class_name: idx for idx, class_name in enumerate(self.config.class_names)}
        
        # 각 분할별 특징 추출
        def extract_split_features(split_files: Dict[str, List[AudioFile]], split_name: str):
            features = []
            labels = []
            
            total_files = sum(len(files) for files in split_files.values())
            processed = 0
            
            for class_name, audio_files in split_files.items():
                class_label = class_to_label[class_name]
                
                for audio_file in audio_files:
                    try:
                        # 특징 추출
                        feature_vector = extract_features(audio_file.file_path, self.config)
                        
                        if feature_vector is not None:
                            features.append(feature_vector.to_array())
                            labels.append(class_label)
                        else:
                            self.logger.warning(f"특징 추출 실패: {audio_file.file_path}")
                    
                    except Exception as e:
                        self.logger.error(f"특징 추출 중 오류 {audio_file.file_path}: {e}")
                        continue
                    
                    processed += 1
                    if processed % 100 == 0:
                        self.logger.info(f"{split_name} 특징 추출 진행: {processed}/{total_files}")
            
            if features:
                X = np.array(features)
                y = np.array(labels)
                self.logger.info(f"{split_name} 특징 추출 완료: {X.shape}")
                return X, y
            else:
                self.logger.error(f"{split_name} 특징 추출 실패: 유효한 특징이 없음")
                return np.array([]), np.array([])
        
        # 훈련 데이터는 증강된 데이터 사용 (이미 augment_training_data 호출했다고 가정)
        if hasattr(self, '_augmented_train_files'):
            train_files = self._augmented_train_files
        else:
            train_files = self._dataset_split.train_files
            self.logger.warning("증강된 훈련 데이터가 없어 원본 데이터 사용")
        
        # 각 분할별 특징 추출
        X_train, y_train = extract_split_features(train_files, "훈련")
        X_val, y_val = extract_split_features(self._dataset_split.validation_files, "검증")
        X_test, y_test = extract_split_features(self._dataset_split.test_files, "테스트")
        
        self.logger.info("=== 전체 특징 추출 완료 ===")
        self.logger.info(f"  훈련: {X_train.shape}")
        self.logger.info(f"  검증: {X_val.shape}")
        self.logger.info(f"  테스트: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def run_complete_pipeline(self, skip_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray, 
                                                                             np.ndarray, np.ndarray, 
                                                                             np.ndarray, np.ndarray]:
        """
        완전한 데이터 파이프라인을 실행합니다.
        
        design.md에 명시된 순서를 따릅니다:
        1. 데이터 로딩 → 2. 소음 파일 검색 → 3. 훈련 증강 → 4. 특징 추출
        
        Parameters:
        -----------
        skip_augmentation : bool
            증강을 건너뛸지 여부
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.logger.info("🍉 완전한 데이터 파이프라인 실행 시작 🍉")
        
        # 1. 데이터 로딩
        dataset_split = self.load_all_data()
        
        # 2. 데이터 무결성 검증
        if not self.validate_data_integrity():
            raise ValueError("데이터 무결성 검증 실패")
        
        # 3. 훈련 증강 (옵션)
        if not skip_augmentation:
            augmented_train = self.augment_training_data()
            self._augmented_train_files = augmented_train
        else:
            self.logger.info("증강 건너뜀")
            self._augmented_train_files = dataset_split.train_files
        
        # 4. 특징 추출
        features = self.extract_all_features()
        
        self.logger.info("🍉 완전한 데이터 파이프라인 실행 완료 🍉")
        
        return features
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        파이프라인 실행 요약을 반환합니다.
        
        Returns:
        --------
        Dict[str, Any]
            파이프라인 실행 요약 정보
        """
        if self._dataset_split is None:
            return {"status": "not_loaded"}
        
        summary = {
            "status": "loaded",
            "data_splits": {
                "train": self._dataset_split.total_train,
                "validation": self._dataset_split.total_validation,
                "test": self._dataset_split.total_test
            },
            "noise_files": len(self._dataset_split.noise_files),
            "augmentation_results": {}
        }
        
        # 증강 결과 포함
        if self._augmentation_results:
            for class_name, result in self._augmentation_results.items():
                summary["augmentation_results"][class_name] = {
                    "original_files": len(self._dataset_split.train_files.get(class_name, [])),
                    "augmented_files": result.total_created,
                    "noise_types_used": result.noise_types_used,
                    "snr_levels_used": result.snr_levels_used
                }
        
        return summary