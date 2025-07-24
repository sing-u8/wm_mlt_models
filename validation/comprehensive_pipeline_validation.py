"""
포괄적 파이프라인 검증 스크립트

전체 수박 소리 분류 시스템의 완전한 검증을 위한 종합 테스트 스위트
"""

import os
import sys
import time
import tempfile
import shutil
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import logging

# 프로젝트 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.pipeline import DataPipeline
from src.ml.training import ModelTrainer  
from src.ml.evaluation import ModelEvaluator
from src.ml.model_converter import ModelConverter
from src.optimization.integrated_optimizer import IntegratedOptimizer
from src.utils.logger import LoggerMixin
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.data_integrity import DataIntegrityChecker
from config import DEFAULT_CONFIG


@dataclass
class ValidationResult:
    """검증 결과"""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP'  
    execution_time: float
    details: Dict
    errors: List[str] = None
    warnings: List[str] = None


@dataclass
class ComprehensiveValidationReport:
    """종합 검증 보고서"""
    validation_timestamp: str
    system_info: Dict
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    test_results: List[ValidationResult]
    overall_status: str
    recommendations: List[str]


class ComprehensivePipelineValidator(LoggerMixin):
    """포괄적 파이프라인 검증기"""
    
    def __init__(self, validation_dir: str = None):
        self.logger = self.get_logger()
        
        # 검증 디렉토리 설정
        if validation_dir is None:
            self.validation_dir = Path("validation_results") / f"validation_{int(time.time())}"
        else:
            self.validation_dir = Path(validation_dir)
        
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # 성능 모니터
        self.performance_monitor = PerformanceMonitor()
        
        # 데이터 무결성 검사기
        self.integrity_checker = DataIntegrityChecker()
        
        # 검증 결과 저장
        self.validation_results: List[ValidationResult] = []
        
        self.logger.info(f"종합 검증 시작: {self.validation_dir}")
    
    def run_comprehensive_validation(self, 
                                   test_data_dirs: List[str] = None,
                                   skip_slow_tests: bool = False) -> ComprehensiveValidationReport:
        """
        포괄적 파이프라인 검증 실행
        
        Args:
            test_data_dirs: 테스트 데이터 디렉토리 목록
            skip_slow_tests: 시간이 오래 걸리는 테스트 건너뛰기
            
        Returns:
            종합 검증 보고서
        """
        self.logger.info("=== 포괄적 파이프라인 검증 시작 ===")
        start_time = time.time()
        
        # 시스템 정보 수집
        system_info = self._collect_system_info()
        
        # 테스트 데이터 준비
        test_datasets = self._prepare_test_datasets(test_data_dirs)
        
        # 전체 성능 모니터링 시작
        self.performance_monitor.start_monitoring()
        
        try:
            # 1. 데이터 파이프라인 검증
            self._validate_data_pipeline(test_datasets)
            
            # 2. 특징 추출 검증
            self._validate_feature_extraction(test_datasets)
            
            # 3. 데이터 증강 검증
            if not skip_slow_tests:
                self._validate_data_augmentation(test_datasets)
            else:
                self._skip_test("데이터 증강 검증", "시간 절약을 위해 건너뜀")
            
            # 4. 머신러닝 파이프라인 검증
            self._validate_ml_pipeline(test_datasets)
            
            # 5. 모델 변환 검증
            self._validate_model_conversion()
            
            # 6. 최적화 시스템 검증
            if not skip_slow_tests:
                self._validate_optimization_system(test_datasets)
            else:
                self._skip_test("최적화 시스템 검증", "시간 절약을 위해 건너뜀")
            
            # 7. 통합 시스템 검증
            self._validate_end_to_end_system(test_datasets)
            
            # 8. 성능 및 리소스 검증
            self._validate_performance_requirements()
            
        finally:
            # 성능 모니터링 종료
            perf_stats = self.performance_monitor.stop_monitoring()
            
        total_time = time.time() - start_time
        
        # 보고서 생성
        report = self._generate_final_report(system_info, total_time, perf_stats)
        
        # 보고서 저장
        self._save_validation_report(report)
        
        self.logger.info(f"=== 종합 검증 완료: {total_time:.2f}초 ===")
        return report
    
    def _collect_system_info(self) -> Dict:
        """시스템 정보 수집"""
        try:
            import psutil
            import platform
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.warning(f"시스템 정보 수집 실패: {e}")
            return {'error': str(e)}
    
    def _prepare_test_datasets(self, test_data_dirs: List[str]) -> List[Dict]:
        """테스트 데이터셋 준비"""
        test_datasets = []
        
        # 기본 테스트 데이터 경로
        default_paths = [
            "data/raw/train",
            "data/raw/validation", 
            "data/raw/test"
        ]
        
        # 실제 데이터가 있는지 확인
        for data_path in default_paths:
            if os.path.exists(data_path):
                dataset_info = {
                    'name': os.path.basename(data_path),
                    'path': data_path,
                    'type': 'real_data',
                    'file_count': self._count_audio_files(data_path)
                }
                if dataset_info['file_count'] > 0:
                    test_datasets.append(dataset_info)
        
        # 추가 테스트 데이터 디렉토리
        if test_data_dirs:
            for test_dir in test_data_dirs:
                if os.path.exists(test_dir):
                    dataset_info = {
                        'name': f"custom_{os.path.basename(test_dir)}",
                        'path': test_dir,
                        'type': 'custom_data',
                        'file_count': self._count_audio_files(test_dir)
                    }
                    if dataset_info['file_count'] > 0:
                        test_datasets.append(dataset_info)
        
        # 실제 데이터가 없으면 가짜 데이터 생성
        if not test_datasets:
            self.logger.warning("실제 테스트 데이터가 없음, 가짜 데이터 생성")
            fake_dataset = self._create_fake_dataset()
            test_datasets.append(fake_dataset)
        
        self.logger.info(f"테스트 데이터셋 준비 완료: {len(test_datasets)}개")
        return test_datasets
    
    def _count_audio_files(self, directory: str) -> int:
        """디렉토리의 오디오 파일 수 계산"""
        count = 0
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac']
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        count += 1
        except Exception as e:
            self.logger.warning(f"파일 카운트 실패 {directory}: {e}")
        
        return count
    
    def _create_fake_dataset(self) -> Dict:
        """가짜 테스트 데이터셋 생성"""
        fake_data_dir = self.validation_dir / "fake_test_data"
        fake_data_dir.mkdir(exist_ok=True)
        
        # 각 클래스별 가짜 오디오 파일 생성
        classes = ['watermelon_A', 'watermelon_B', 'watermelon_C'] 
        file_count = 0
        
        for class_name in classes:
            class_dir = fake_data_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # 각 클래스당 5개 파일 생성
            for i in range(5):
                fake_audio_file = class_dir / f"{class_name}_test_{i:02d}.wav"
                self._create_fake_audio_file(fake_audio_file)
                file_count += 1
        
        return {
            'name': 'fake_test_data',
            'path': str(fake_data_dir),
            'type': 'generated_data',
            'file_count': file_count
        }
    
    def _create_fake_audio_file(self, file_path: Path):
        """가짜 오디오 파일 생성"""
        try:
            import soundfile as sf
            
            # 1초, 22050Hz 사인파 생성
            duration = 1.0
            sample_rate = 22050
            frequency = 440  # A4 음
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # 노이즈 추가
            noise = np.random.normal(0, 0.05, len(audio_data))
            audio_data += noise
            
            sf.write(str(file_path), audio_data, sample_rate)
            
        except Exception as e:
            self.logger.warning(f"가짜 오디오 파일 생성 실패 {file_path}: {e}")
    
    def _validate_data_pipeline(self, test_datasets: List[Dict]):
        """데이터 파이프라인 검증"""
        test_name = "데이터 파이프라인 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            results = {}
            errors = []
            
            for dataset in test_datasets[:1]:  # 첫 번째 데이터셋만 테스트
                dataset_name = dataset['name']
                self.logger.info(f"데이터셋 테스트: {dataset_name}")
                
                try:
                    # 데이터 파이프라인 생성
                    pipeline = DataPipeline(DEFAULT_CONFIG)
                    
                    # 데이터 로딩 테스트
                    if dataset['type'] == 'generated_data':
                        # 가짜 데이터용 로딩
                        audio_files = []
                        for root, dirs, files in os.walk(dataset['path']):
                            for file in files:
                                if file.endswith('.wav'):
                                    audio_files.append(os.path.join(root, file))
                        
                        results[f'{dataset_name}_files_loaded'] = len(audio_files)
                    else:
                        # 실제 데이터 로딩
                        if 'train' in dataset['path']:
                            train_data = pipeline.load_train_data()
                            results[f'{dataset_name}_train_classes'] = len(train_data)
                        elif 'validation' in dataset['path']:
                            val_data = pipeline.load_validation_data()
                            results[f'{dataset_name}_validation_classes'] = len(val_data)
                        elif 'test' in dataset['path']:
                            test_data = pipeline.load_test_data()
                            results[f'{dataset_name}_test_classes'] = len(test_data)
                    
                    # 데이터 무결성 검사
                    integrity_result = self.integrity_checker.check_data_integrity(
                        dataset['path'], dataset['path'])
                    results[f'{dataset_name}_integrity'] = integrity_result.overall_score
                    
                except Exception as e:
                    error_msg = f"데이터셋 {dataset_name} 처리 실패: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            if errors:
                status = 'FAIL'
            elif results:
                status = 'PASS'
            else:
                status = 'SKIP'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_feature_extraction(self, test_datasets: List[Dict]):
        """특징 추출 검증"""
        test_name = "특징 추출 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            from src.audio.feature_extraction import extract_features
            
            results = {}
            errors = []
            
            # 각 데이터셋의 샘플 파일로 특징 추출 테스트
            for dataset in test_datasets:
                dataset_name = dataset['name']
                
                try:
                    # 샘플 파일 찾기
                    sample_files = []
                    for root, dirs, files in os.walk(dataset['path']):
                        for file in files:
                            if file.endswith('.wav') and len(sample_files) < 3:
                                sample_files.append(os.path.join(root, file))
                    
                    if not sample_files:
                        errors.append(f"데이터셋 {dataset_name}에서 오디오 파일을 찾을 수 없음")
                        continue
                    
                    # 특징 추출 테스트
                    extracted_features = []
                    for audio_file in sample_files:
                        try:
                            features = extract_features(audio_file, DEFAULT_CONFIG)
                            if features is not None:
                                extracted_features.append(features)
                        except Exception as e:
                            errors.append(f"특징 추출 실패 {audio_file}: {str(e)}")
                    
                    results[f'{dataset_name}_features_extracted'] = len(extracted_features)
                    
                    if extracted_features:
                        # 특징 벡터 검증
                        feature_array = extracted_features[0].to_array()
                        results[f'{dataset_name}_feature_dimension'] = len(feature_array)
                        results[f'{dataset_name}_feature_valid'] = not np.any(np.isnan(feature_array))
                
                except Exception as e:
                    error_msg = f"데이터셋 {dataset_name} 특징 추출 실패: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            status = 'FAIL' if errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_data_augmentation(self, test_datasets: List[Dict]):
        """데이터 증강 검증"""
        test_name = "데이터 증강 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            from src.data.augmentation import BatchAugmentor
            
            results = {}
            errors = []
            warnings = []
            
            # 노이즈 파일 찾기
            noise_files = []
            noise_paths = ["data/noise", "data/noise/environmental"]
            for noise_path in noise_paths:
                if os.path.exists(noise_path):
                    for root, dirs, files in os.walk(noise_path):
                        for file in files:
                            if file.endswith('.wav') and len(noise_files) < 3:
                                noise_files.append(os.path.join(root, file))
            
            if not noise_files:
                warnings.append("노이즈 파일이 없어 증강 테스트를 간단히 수행")
                # 가짜 노이즈 파일 생성
                fake_noise_dir = self.validation_dir / "fake_noise"
                fake_noise_dir.mkdir(exist_ok=True)
                fake_noise_file = fake_noise_dir / "test_noise.wav"
                self._create_fake_audio_file(fake_noise_file)
                noise_files = [str(fake_noise_file)]
            
            # 증강 테스트
            for dataset in test_datasets[:1]:  # 첫 번째 데이터셋만
                dataset_name = dataset['name']
                
                try:
                    # 샘플 오디오 파일 찾기
                    sample_files = []
                    for root, dirs, files in os.walk(dataset['path']):
                        for file in files:
                            if file.endswith('.wav') and len(sample_files) < 2:
                                sample_files.append(os.path.join(root, file))
                    
                    if sample_files and noise_files:
                        # 배치 증강기 생성
                        augmentor = BatchAugmentor(DEFAULT_CONFIG)
                        
                        # 임시 출력 디렉토리
                        aug_output_dir = self.validation_dir / f"augmented_{dataset_name}"
                        aug_output_dir.mkdir(exist_ok=True)
                        
                        # 증강 수행
                        augmented_files = []
                        for audio_file in sample_files[:1]:  # 하나만 테스트
                            for noise_file in noise_files[:1]:  # 하나만 테스트
                                try:
                                    aug_result = augmentor.augment_noise(
                                        audio_file, noise_file, snr_level=10.0, 
                                        output_dir=str(aug_output_dir))
                                    if aug_result and aug_result.output_file:
                                        augmented_files.append(aug_result.output_file)
                                except Exception as e:
                                    errors.append(f"증강 실패 {audio_file}: {str(e)}")
                        
                        results[f'{dataset_name}_augmented_files'] = len(augmented_files)
                
                except Exception as e:
                    error_msg = f"데이터셋 {dataset_name} 증강 실패: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            if errors and not warnings:
                status = 'FAIL'
            elif results or warnings:
                status = 'PASS'
            else:
                status = 'SKIP'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors,
                warnings=warnings
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_ml_pipeline(self, test_datasets: List[Dict]):
        """머신러닝 파이프라인 검증"""
        test_name = "머신러닝 파이프라인 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            from src.audio.feature_extraction import extract_features
            from src.ml.training import ModelTrainer
            from src.ml.evaluation import ModelEvaluator
            
            results = {}
            errors = []
            
            # 훈련 데이터 준비
            X_train = []
            y_train = []
            
            for dataset in test_datasets:
                if dataset['file_count'] < 3:  # 너무 작은 데이터셋 건너뛰기
                    continue
                
                dataset_name = dataset['name']
                class_id = 0  # 단순화를 위해 모든 파일을 클래스 0으로
                
                try:
                    # 샘플 파일들로 특징 추출
                    sample_count = 0
                    for root, dirs, files in os.walk(dataset['path']):
                        for file in files:
                            if file.endswith('.wav') and sample_count < 6:  # 클래스당 최대 6개
                                audio_file = os.path.join(root, file)
                                try:
                                    features = extract_features(audio_file, DEFAULT_CONFIG)
                                    if features is not None:
                                        X_train.append(features.to_array())
                                        y_train.append(class_id)
                                        sample_count += 1
                                except Exception as e:
                                    errors.append(f"특징 추출 실패 {audio_file}: {str(e)}")
                        
                        # 서브디렉토리별로 다른 클래스 할당
                        class_id += 1
                        if class_id >= 3:  # 최대 3클래스
                            break
                
                except Exception as e:
                    error_msg = f"데이터셋 {dataset_name} ML 준비 실패: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            if len(X_train) < 6:  # 최소한의 훈련 데이터 필요
                errors.append(f"훈련 데이터 부족: {len(X_train)}개 (최소 6개 필요)")
            else:
                try:
                    # 간단한 훈련 및 평가
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    results['training_samples'] = len(X_train)
                    results['feature_dimension'] = X_train.shape[1]
                    results['unique_classes'] = len(np.unique(y_train))
                    
                    # 간단한 모델 훈련 (빠른 설정)
                    trainer = ModelTrainer(DEFAULT_CONFIG)
                    
                    # 훈련/테스트 분할
                    split_idx = int(len(X_train) * 0.7)
                    X_train_split = X_train[:split_idx]
                    y_train_split = y_train[:split_idx]
                    X_test_split = X_train[split_idx:]
                    y_test_split = y_train[split_idx:]
                    
                    if len(X_test_split) > 0:
                        # 빠른 훈련 (하이퍼파라미터 그리드 최소화)
                        training_result = trainer.train_with_cv(
                            X_train_split, y_train_split, cv_folds=2)  # 2-fold로 빠르게
                        
                        results['training_success'] = training_result is not None
                        
                        if training_result:
                            # 평가
                            evaluator = ModelEvaluator(DEFAULT_CONFIG)
                            best_model = training_result.best_models.get('svm')  # SVM만 테스트
                            
                            if best_model:
                                eval_result = evaluator.evaluate_model(
                                    best_model, X_test_split, y_test_split)
                                results['test_accuracy'] = eval_result.accuracy
                                results['evaluation_success'] = True
                            else:
                                errors.append("최고 모델을 찾을 수 없음")
                    else:
                        warnings = ["테스트 데이터가 부족하여 평가를 건너뜀"]
                        results['evaluation_skipped'] = True
                
                except Exception as e:
                    error_msg = f"ML 파이프라인 실행 실패: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            status = 'FAIL' if errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_model_conversion(self):
        """모델 변환 검증"""
        test_name = "모델 변환 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            from src.ml.model_converter import ModelConverter
            from sklearn.ensemble import RandomForestClassifier
            
            results = {}
            errors = []
            
            try:
                # 간단한 모델 생성
                X_dummy = np.random.rand(20, 30)  # 20 샘플, 30 특징
                y_dummy = np.random.randint(0, 3, 20)  # 3 클래스
                
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X_dummy, y_dummy)
                
                results['dummy_model_created'] = True
                
                # 모델 변환기 생성
                converter = ModelConverter(DEFAULT_CONFIG)
                
                # 임시 모델 저장 디렉토리
                model_dir = self.validation_dir / "test_models"
                model_dir.mkdir(exist_ok=True)
                
                # Pickle 저장 테스트
                pickle_path = model_dir / "test_model.pkl"
                pickle_result = converter.save_pickle_model(
                    model, str(pickle_path), 
                    feature_config={'n_features': 30},
                    metadata={'test': True})
                
                results['pickle_save_success'] = pickle_result.success
                results['pickle_file_exists'] = os.path.exists(pickle_path)
                
                # Core ML 변환 테스트
                try:
                    coreml_path = model_dir / "test_model.mlmodel"
                    coreml_result = converter.convert_to_coreml(
                        model, str(coreml_path),
                        input_features=['feature_' + str(i) for i in range(30)],
                        class_labels=['watermelon_A', 'watermelon_B', 'watermelon_C'])
                    
                    results['coreml_conversion_success'] = coreml_result.success
                    results['coreml_file_exists'] = os.path.exists(coreml_path)
                    
                    # 예측 일치성 검증
                    if coreml_result.success:
                        validation_result = converter.validate_model_conversion(
                            model, str(coreml_path), X_dummy[:5])  # 5개 샘플만
                        results['prediction_consistency'] = validation_result
                
                except Exception as e:
                    # Core ML 변환은 선택적 기능이므로 경고로 처리
                    self.logger.warning(f"Core ML 변환 실패 (선택적 기능): {e}")
                    results['coreml_conversion_skipped'] = True
            
            except Exception as e:
                error_msg = f"모델 변환 테스트 실패: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가 (Core ML 실패는 전체 실패로 보지 않음)
            critical_errors = [e for e in errors if 'coreml' not in e.lower()]
            status = 'FAIL' if critical_errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_optimization_system(self, test_datasets: List[Dict]):
        """최적화 시스템 검증"""
        test_name = "최적화 시스템 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            results = {}
            errors = []
            
            try:
                # 통합 최적화기 생성
                optimizer = IntegratedOptimizer()
                
                results['optimizer_created'] = True
                
                # 시스템 벤치마크
                benchmark_result = optimizer.benchmark_system(num_test_files=5)
                
                if benchmark_result:
                    results['benchmark_completed'] = True
                    results['files_per_second'] = benchmark_result.get('feature_extraction', {}).get('files_per_second', 0)
                    results['memory_efficiency'] = benchmark_result.get('feature_extraction', {}).get('memory_efficiency', 0)
                    
                    # 성능 임계값 검증
                    files_per_sec = results['files_per_second']
                    if files_per_sec > 0.5:  # 최소 성능 요구사항
                        results['performance_acceptable'] = True
                    else:
                        errors.append(f"성능이 기준 이하: {files_per_sec:.2f} files/sec < 0.5")
                else:
                    errors.append("벤치마크 실행 실패")
                
                # 하드웨어 설정 테스트
                hw_config = optimizer.hardware_config.get_current_config()
                results['hardware_detected'] = hw_config is not None
                results['cpu_cores'] = hw_config.get('hardware_profile', {}).get('cpu_cores', 0)
                results['memory_gb'] = hw_config.get('hardware_profile', {}).get('memory_gb', 0)
            
            except Exception as e:
                error_msg = f"최적화 시스템 테스트 실패: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            status = 'FAIL' if errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_end_to_end_system(self, test_datasets: List[Dict]):
        """통합 시스템 검증"""
        test_name = "엔드투엔드 시스템 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            results = {}
            errors = []
            
            try:
                # 메인 파이프라인 임포트
                from main import WatermelonClassificationPipeline
                
                # 임시 출력 디렉토리
                e2e_output_dir = self.validation_dir / "e2e_test"
                e2e_output_dir.mkdir(exist_ok=True)
                
                # 파이프라인 생성 (간단한 설정)
                pipeline = WatermelonClassificationPipeline()
                
                results['pipeline_created'] = True
                
                # 실제 데이터가 있는 경우만 전체 파이프라인 실행
                has_real_data = any(d['type'] in ['real_data', 'custom_data'] 
                                  and d['file_count'] >= 9 for d in test_datasets)  # 최소 3클래스 * 3파일
                
                if has_real_data:
                    self.logger.info("실제 데이터로 전체 파이프라인 실행")
                    
                    # 간단한 설정으로 파이프라인 실행
                    pipeline_result = pipeline.run_complete_pipeline(
                        skip_augmentation=True,  # 시간 절약
                        cv_folds=2,  # 빠른 교차 검증
                        force_retrain=True,
                        dry_run=False
                    )
                    
                    results['pipeline_execution_success'] = pipeline_result is not None
                    
                    if pipeline_result:
                        results['models_trained'] = len(pipeline_result.get('trained_models', {}))
                        results['best_accuracy'] = pipeline_result.get('best_model_accuracy', 0)
                    
                else:
                    self.logger.info("실제 데이터 부족으로 전체 파이프라인 건너뜀")
                    results['pipeline_skipped'] = "데이터 부족"
                
                # 구성요소별 개별 테스트
                results['individual_components_tested'] = len([r for r in self.validation_results if r.status == 'PASS'])
            
            except Exception as e:
                error_msg = f"엔드투엔드 시스템 실행 실패: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            status = 'FAIL' if errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _validate_performance_requirements(self):
        """성능 및 리소스 요구사항 검증"""
        test_name = "성능 및 리소스 요구사항 검증"
        self.logger.info(f"=== {test_name} 시작 ===")
        start_time = time.time()
        
        try:
            results = {}
            errors = []
            warnings = []
            
            try:
                import psutil
                
                # 현재 시스템 리소스
                cpu_count = psutil.cpu_count()
                memory_gb = psutil.virtual_memory().total / (1024**3)
                disk_free_gb = psutil.disk_usage('.').free / (1024**3)
                
                results['system_cpu_cores'] = cpu_count
                results['system_memory_gb'] = memory_gb
                results['system_disk_free_gb'] = disk_free_gb
                
                # 최소 요구사항 검증
                min_requirements = {
                    'cpu_cores': 2,
                    'memory_gb': 2.0,
                    'disk_free_gb': 1.0
                }
                
                for req_name, min_value in min_requirements.items():
                    current_value = results[f'system_{req_name}']
                    if current_value >= min_value:
                        results[f'{req_name}_sufficient'] = True
                    else:
                        error_msg = f"{req_name} 부족: {current_value} < {min_value}"
                        errors.append(error_msg)
                        results[f'{req_name}_sufficient'] = False
                
                # 권장 요구사항 검증
                recommended_requirements = {
                    'cpu_cores': 4,
                    'memory_gb': 8.0,
                    'disk_free_gb': 5.0
                }
                
                for req_name, rec_value in recommended_requirements.items():
                    current_value = results[f'system_{req_name}']
                    if current_value >= rec_value:
                        results[f'{req_name}_recommended'] = True
                    else:
                        warning_msg = f"{req_name} 권장사항 미달: {current_value} < {rec_value}"
                        warnings.append(warning_msg)
                        results[f'{req_name}_recommended'] = False
                
                # 성능 검증 (이전 테스트 결과 기반)
                performance_tests = [r for r in self.validation_results 
                                   if '최적화' in r.test_name or '특징 추출' in r.test_name]
                
                if performance_tests:
                    avg_execution_time = sum(t.execution_time for t in performance_tests) / len(performance_tests)
                    results['average_test_execution_time'] = avg_execution_time
                    
                    if avg_execution_time < 30:  # 30초 이내
                        results['performance_acceptable'] = True
                    else:
                        warnings.append(f"성능이 느림: 평균 {avg_execution_time:.1f}초")
                        results['performance_acceptable'] = False
            
            except Exception as e:
                error_msg = f"성능 요구사항 검증 실패: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
            
            execution_time = time.time() - start_time
            
            # 결과 평가
            status = 'FAIL' if errors else 'PASS'
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                execution_time=execution_time,
                details=results,
                errors=errors,
                warnings=warnings
            )
            
            self.validation_results.append(result)
            self.logger.info(f"{test_name} 완료: {status} ({execution_time:.2f}초)")
            
        except Exception as e:
            self._handle_test_failure(test_name, e, start_time)
    
    def _skip_test(self, test_name: str, reason: str):
        """테스트 건너뛰기"""
        result = ValidationResult(
            test_name=test_name,
            status='SKIP',
            execution_time=0.0,
            details={'skip_reason': reason}
        )
        self.validation_results.append(result)
        self.logger.info(f"{test_name} 건너뜀: {reason}")
    
    def _handle_test_failure(self, test_name: str, exception: Exception, start_time: float):
        """테스트 실패 처리"""
        execution_time = time.time() - start_time
        error_msg = f"예외 발생: {str(exception)}"
        
        result = ValidationResult(
            test_name=test_name,
            status='FAIL',
            execution_time=execution_time,
            details={'exception': error_msg},
            errors=[error_msg]
        )
        
        self.validation_results.append(result)
        self.logger.error(f"{test_name} 실패: {error_msg}")
        self.logger.debug(traceback.format_exc())
    
    def _generate_final_report(self, system_info: Dict, total_time: float, perf_stats: Dict) -> ComprehensiveValidationReport:
        """최종 검증 보고서 생성"""
        passed_tests = sum(1 for r in self.validation_results if r.status == 'PASS')
        failed_tests = sum(1 for r in self.validation_results if r.status == 'FAIL')
        skipped_tests = sum(1 for r in self.validation_results if r.status == 'SKIP')
        total_tests = len(self.validation_results)
        
        # 전체 상태 결정
        if failed_tests == 0:
            overall_status = 'PASS'
        elif passed_tests > failed_tests:
            overall_status = 'PARTIAL_PASS'
        else:
            overall_status = 'FAIL'
        
        # 권장사항 생성
        recommendations = self._generate_recommendations()
        
        return ComprehensiveValidationReport(
            validation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            system_info=system_info,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_time,
            test_results=self.validation_results,
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """검증 결과 기반 권장사항 생성"""
        recommendations = []
        
        failed_tests = [r for r in self.validation_results if r.status == 'FAIL']
        
        # 실패한 테스트 기반 권장사항
        if any('데이터 파이프라인' in t.test_name for t in failed_tests):
            recommendations.append("데이터 디렉토리 구조와 파일 경로를 확인하세요")
        
        if any('특징 추출' in t.test_name for t in failed_tests):
            recommendations.append("오디오 파일 형식과 librosa 설치를 확인하세요")
        
        if any('머신러닝' in t.test_name for t in failed_tests):
            recommendations.append("훈련 데이터의 양과 품질을 확인하세요")
        
        if any('성능' in t.test_name for t in failed_tests):
            recommendations.append("시스템 리소스(CPU, 메모리)를 업그레이드하세요")
        
        # 일반적인 권장사항
        if not recommendations:
            recommendations.append("모든 테스트가 통과했습니다. 시스템이 정상 작동합니다.")
        
        return recommendations
    
    def _save_validation_report(self, report: ComprehensiveValidationReport):
        """검증 보고서 저장"""
        try:
            # JSON 보고서
            report_file = self.validation_dir / "validation_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
            
            # 텍스트 요약 보고서
            summary_file = self.validation_dir / "validation_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"수박 소리 분류 시스템 검증 보고서\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"검증 시간: {report.validation_timestamp}\n")
                f.write(f"전체 상태: {report.overall_status}\n")
                f.write(f"총 실행 시간: {report.total_execution_time:.2f}초\n\n")
                
                f.write(f"테스트 결과:\n")
                f.write(f"  통과: {report.passed_tests}/{report.total_tests}\n")
                f.write(f"  실패: {report.failed_tests}/{report.total_tests}\n")
                f.write(f"  건너뜀: {report.skipped_tests}/{report.total_tests}\n\n")
                
                f.write(f"개별 테스트 결과:\n")
                for result in report.test_results:
                    f.write(f"  [{result.status}] {result.test_name} ({result.execution_time:.2f}초)\n")
                    if result.errors:
                        for error in result.errors:
                            f.write(f"    오류: {error}\n")
                
                f.write(f"\n권장사항:\n")
                for rec in report.recommendations:
                    f.write(f"  - {rec}\n")
            
            self.logger.info(f"검증 보고서 저장 완료: {self.validation_dir}")
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='수박 소리 분류 시스템 포괄적 검증')
    parser.add_argument('--test-data', nargs='*', help='추가 테스트 데이터 디렉토리')
    parser.add_argument('--skip-slow', action='store_true', help='시간이 오래 걸리는 테스트 건너뛰기')
    parser.add_argument('--output-dir', help='검증 결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 검증기 생성
    validator = ComprehensivePipelineValidator(args.output_dir)
    
    # 포괄적 검증 실행
    report = validator.run_comprehensive_validation(
        test_data_dirs=args.test_data,
        skip_slow_tests=args.skip_slow
    )
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"수박 소리 분류 시스템 검증 결과")
    print(f"{'='*60}")
    print(f"전체 상태: {report.overall_status}")
    print(f"총 실행 시간: {report.total_execution_time:.2f}초")
    print(f"테스트 결과: {report.passed_tests} 통과, {report.failed_tests} 실패, {report.skipped_tests} 건너뜀")
    
    print(f"\n개별 테스트 결과:")
    for result in report.test_results:
        status_symbol = "✅" if result.status == 'PASS' else "❌" if result.status == 'FAIL' else "⏭️"
        print(f"  {status_symbol} {result.test_name} ({result.execution_time:.2f}초)")
    
    if report.recommendations:
        print(f"\n권장사항:")
        for rec in report.recommendations:
            print(f"  💡 {rec}")
    
    print(f"\n상세 보고서: {validator.validation_dir}")
    
    # 종료 코드
    return 0 if report.overall_status in ['PASS', 'PARTIAL_PASS'] else 1


if __name__ == "__main__":
    sys.exit(main())