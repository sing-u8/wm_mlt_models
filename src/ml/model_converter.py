"""
모델 형식 변환 모듈

이 모듈은 훈련된 scikit-learn 모델을 다양한 형식으로 변환하고
배포를 위한 모델 아티팩트를 관리합니다.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

try:
    import coremltools as ct
    from coremltools.models import MLModel
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from ..utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class ConversionResult:
    """
    모델 변환 결과를 담는 데이터 클래스.
    """
    model_name: str
    original_format: str
    target_format: str
    original_path: str
    converted_path: str
    conversion_time: float
    validation_passed: bool
    validation_details: Dict[str, Any]
    created_at: str
    file_size_bytes: int


@dataclass
class CoreMLModelInfo:
    """
    Core ML 모델 정보를 담는 데이터 클래스.
    """
    model_path: str
    input_name: str
    input_shape: Tuple[int, ...]
    output_name: str
    class_labels: List[str]
    model_version: str
    description: str
    author: str = "Watermelon Sound Classifier"
    
    def to_metadata(self) -> Dict[str, Any]:
        """메타데이터를 딕셔너리로 변환"""
        return asdict(self)


class ModelConverter(LoggerMixin):
    """
    모델 형식 변환을 위한 클래스.
    
    design.md에 명시된 인터페이스를 구현합니다.
    """
    
    def __init__(self, config=None):
        """
        변환 구성으로 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        self.conversion_results = {}
        
        # Core ML 사용 가능성 확인
        if not COREML_AVAILABLE:
            self.logger.warning("coremltools가 설치되지 않았습니다. Core ML 변환을 사용할 수 없습니다.")
            self.logger.warning("설치하려면: pip install coremltools")
        
        self.logger.info(f"ModelConverter 초기화됨 (Core ML 지원: {COREML_AVAILABLE})")
    
    def save_pickle_model(self, model: Any, model_name: str, 
                         model_metadata: Dict[str, Any] = None,
                         output_dir: str = None) -> str:
        """
        모델을 pickle 형식으로 저장합니다.
        
        Parameters:
        -----------
        model : Any
            저장할 scikit-learn 모델
        model_name : str
            모델 이름
        model_metadata : Dict[str, Any], optional
            모델 메타데이터
        output_dir : str, optional
            출력 디렉토리
            
        Returns:
        --------
        str
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.model_output_dir, "pickle")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 파일 경로
        model_filename = f"{model_name}_model.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        start_time = datetime.now()
        
        try:
            # 모델 저장
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 메타데이터 저장
            if model_metadata:
                metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
                
                # 모델 정보 추가
                full_metadata = {
                    **model_metadata,
                    'model_path': model_path,
                    'model_format': 'pickle',
                    'created_at': datetime.now().isoformat(),
                    'model_size_bytes': os.path.getsize(model_path),
                    'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                    'sklearn_version': __import__('sklearn').__version__
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(full_metadata, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"메타데이터 저장됨: {metadata_path}")
            
            save_time = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(model_path)
            
            self.logger.info(f"✅ Pickle 모델 저장 완료:")
            self.logger.info(f"  파일: {model_path}")
            self.logger.info(f"  크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            self.logger.info(f"  저장 시간: {save_time:.3f}초")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Pickle 모델 저장 실패: {e}")
            raise
    
    def load_pickle_model(self, model_path: str) -> Any:
        """
        Pickle 모델을 로드합니다.
        
        Parameters:
        -----------
        model_path : str
            모델 파일 경로
            
        Returns:
        --------
        Any
            로드된 모델 객체
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"Pickle 모델 로드 완료: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Pickle 모델 로드 실패 {model_path}: {e}")
            raise
    
    def convert_to_coreml(self, model, model_name: str, 
                         input_description: Dict[str, Any] = None,
                         output_dir: str = None) -> str:
        """
        scikit-learn 모델을 Core ML 형식으로 변환합니다.
        
        Parameters:
        -----------
        model : sklearn model
            변환할 scikit-learn 모델
        model_name : str
            모델 이름
        input_description : Dict[str, Any], optional
            입력 설명 정보
        output_dir : str, optional
            출력 디렉토리
            
        Returns:
        --------
        str
            변환된 Core ML 모델 파일 경로
        """
        if not COREML_AVAILABLE:
            raise ImportError("coremltools가 설치되지 않았습니다. 'pip install coremltools'로 설치하세요.")
        
        if output_dir is None:
            output_dir = os.path.join(self.config.model_output_dir, "coreml")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"=== {model_name} Core ML 변환 시작 ===")
        
        start_time = datetime.now()
        
        try:
            # 입력 설명 설정
            if input_description is None:
                input_description = {
                    'name': 'audio_features',
                    'shape': (30,),  # design.md 명세: 30차원 특징 벡터
                    'description': 'Audio feature vector extracted from watermelon sound'
                }
            
            # 샘플 입력 데이터 생성 (변환을 위해 필요)
            sample_input = np.random.randn(1, input_description['shape'][0]).astype(np.float32)
            
            # Core ML 모델로 변환
            self.logger.info("Core ML 변환 중...")
            
            # scikit-learn 모델을 Core ML로 변환
            coreml_model = ct.converters.sklearn.convert(
                model,
                input_features=[input_description['name']],
                class_labels=self.config.class_names,
                output_feature_names=['watermelon_class', 'watermelon_class_proba']
            )
            
            # 모델 메타데이터 설정
            coreml_model.short_description = f"Watermelon ripeness classifier - {model_name}"
            coreml_model.author = "Watermelon Sound Classifier System"
            coreml_model.license = "MIT"
            coreml_model.version = self.config.model_version if hasattr(self.config, 'model_version') else "1.0"
            
            # 입력/출력 설명 추가
            coreml_model.input_description[input_description['name']] = input_description.get(
                'description', 'Audio feature vector'
            )
            coreml_model.output_description['watermelon_class'] = 'Predicted watermelon ripeness class'
            coreml_model.output_description['watermelon_class_proba'] = 'Class prediction probabilities'
            
            # 파일 저장
            coreml_filename = f"{model_name}_model.mlmodel"
            coreml_path = os.path.join(output_dir, coreml_filename)
            
            coreml_model.save(coreml_path)
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(coreml_path)
            
            # Core ML 모델 정보 생성
            coreml_info = CoreMLModelInfo(
                model_path=coreml_path,
                input_name=input_description['name'],
                input_shape=input_description['shape'],
                output_name='watermelon_class',
                class_labels=self.config.class_names,
                model_version=coreml_model.version,
                description=coreml_model.short_description
            )
            
            # 메타데이터 저장
            metadata_path = os.path.join(output_dir, f"{model_name}_coreml_info.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(coreml_info.to_metadata(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Core ML 변환 완료:")
            self.logger.info(f"  파일: {coreml_path}")
            self.logger.info(f"  크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            self.logger.info(f"  변환 시간: {conversion_time:.3f}초")
            self.logger.info(f"  입력: {input_description['name']} {input_description['shape']}")
            self.logger.info(f"  출력: watermelon_class (클래스), watermelon_class_proba (확률)")
            
            return coreml_path
            
        except Exception as e:
            self.logger.error(f"Core ML 변환 실패: {e}")
            raise
    
    def validate_model_conversion(self, original_model, coreml_path: str,
                                test_samples: int = 10) -> Dict[str, Any]:
        """
        변환된 Core ML 모델의 예측 결과를 원본 모델과 비교하여 검증합니다.
        
        Parameters:
        -----------
        original_model : sklearn model
            원본 scikit-learn 모델
        coreml_path : str
            Core ML 모델 파일 경로
        test_samples : int
            검증에 사용할 테스트 샘플 수
            
        Returns:
        --------
        Dict[str, Any]
            검증 결과
        """
        if not COREML_AVAILABLE:
            return {'error': 'coremltools not available'}
        
        self.logger.info(f"=== 모델 변환 검증 시작 ({test_samples}개 샘플) ===")
        
        try:
            # Core ML 모델 로드
            coreml_model = ct.models.MLModel(coreml_path)
            
            # 테스트 데이터 생성
            np.random.seed(42)
            test_data = np.random.randn(test_samples, 30).astype(np.float32)
            
            # 원본 모델 예측
            original_predictions = original_model.predict(test_data)
            original_probabilities = None
            if hasattr(original_model, 'predict_proba'):
                original_probabilities = original_model.predict_proba(test_data)
            
            # Core ML 모델 예측
            coreml_predictions = []
            coreml_probabilities = []
            
            for i in range(test_samples):
                # 입력 데이터 준비
                input_dict = {'audio_features': test_data[i:i+1]}
                
                # 예측 수행
                coreml_result = coreml_model.predict(input_dict)
                
                # 클래스 예측 추출
                if 'watermelon_class' in coreml_result:
                    class_pred = coreml_result['watermelon_class']
                    if isinstance(class_pred, (list, np.ndarray)):
                        class_pred = class_pred[0]
                    
                    # 클래스 이름을 인덱스로 변환
                    if isinstance(class_pred, str):
                        class_idx = self.config.class_names.index(class_pred)
                        coreml_predictions.append(class_idx)
                    else:
                        coreml_predictions.append(int(class_pred))
                
                # 확률 예측 추출
                if 'watermelon_class_proba' in coreml_result:
                    proba = coreml_result['watermelon_class_proba']
                    if isinstance(proba, dict):
                        # 딕셔너리 형태인 경우 클래스 순서대로 정렬
                        proba_array = [proba.get(class_name, 0.0) for class_name in self.config.class_names]
                        coreml_probabilities.append(proba_array)
                    else:
                        coreml_probabilities.append(proba)
            
            coreml_predictions = np.array(coreml_predictions)
            if coreml_probabilities:
                coreml_probabilities = np.array(coreml_probabilities)
            
            # 예측 결과 비교
            class_accuracy = np.mean(original_predictions == coreml_predictions)
            
            validation_result = {
                'test_samples': test_samples,
                'class_prediction_accuracy': float(class_accuracy),
                'predictions_match': bool(class_accuracy == 1.0),
                'original_predictions': original_predictions.tolist(),
                'coreml_predictions': coreml_predictions.tolist()
            }
            
            # 확률 예측 비교 (가능한 경우)
            if original_probabilities is not None and len(coreml_probabilities) > 0:
                prob_diff = np.abs(original_probabilities - coreml_probabilities)
                max_prob_diff = np.max(prob_diff)
                mean_prob_diff = np.mean(prob_diff)
                
                validation_result.update({
                    'probability_validation': True,
                    'max_probability_difference': float(max_prob_diff),
                    'mean_probability_difference': float(mean_prob_diff),
                    'probabilities_close': bool(max_prob_diff < 0.01)  # 1% 이내
                })
            
            # 결과 로깅
            self.logger.info(f"✅ 모델 변환 검증 완료:")
            self.logger.info(f"  클래스 예측 정확도: {class_accuracy:.1%}")
            
            if 'probability_validation' in validation_result:
                self.logger.info(f"  확률 예측 최대 차이: {validation_result['max_probability_difference']:.6f}")
                self.logger.info(f"  확률 예측 평균 차이: {validation_result['mean_probability_difference']:.6f}")
            
            if validation_result['predictions_match']:
                self.logger.info("  ✅ 모든 예측이 일치합니다!")
            else:
                self.logger.warning(f"  ⚠️ {test_samples - int(class_accuracy * test_samples)}개 예측이 불일치합니다.")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"모델 변환 검증 실패: {e}")
            return {
                'error': str(e),
                'validation_passed': False
            }
    
    def convert_model_with_validation(self, model, model_name: str,
                                    input_description: Dict[str, Any] = None,
                                    output_dir: str = None,
                                    validate: bool = True) -> ConversionResult:
        """
        모델을 Core ML로 변환하고 검증을 수행합니다.
        
        Parameters:
        -----------
        model : sklearn model
            변환할 모델
        model_name : str
            모델 이름
        input_description : Dict[str, Any], optional
            입력 설명
        output_dir : str, optional
            출력 디렉토리
        validate : bool
            변환 후 검증 수행 여부
            
        Returns:
        --------
        ConversionResult
            변환 결과
        """
        start_time = datetime.now()
        
        try:
            # Core ML 변환
            coreml_path = self.convert_to_coreml(
                model, model_name, input_description, output_dir
            )
            
            # 검증 수행
            validation_result = {'validation_performed': False}
            validation_passed = True
            
            if validate:
                validation_result = self.validate_model_conversion(model, coreml_path)
                validation_passed = validation_result.get('predictions_match', False)
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(coreml_path)
            
            # 변환 결과 생성
            result = ConversionResult(
                model_name=model_name,
                original_format='sklearn',
                target_format='coreml',
                original_path='N/A',  # 메모리의 모델
                converted_path=coreml_path,
                conversion_time=conversion_time,
                validation_passed=validation_passed,
                validation_details=validation_result,
                created_at=datetime.now().isoformat(),
                file_size_bytes=file_size
            )
            
            # 결과 저장
            self.conversion_results[model_name] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"모델 변환 및 검증 실패: {e}")
            raise
    
    def create_model_metadata(self, model_info: Dict[str, Any],
                            training_info: Dict[str, Any] = None,
                            performance_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        모델 메타데이터를 생성합니다.
        
        Parameters:
        -----------
        model_info : Dict[str, Any]
            기본 모델 정보
        training_info : Dict[str, Any], optional
            훈련 정보
        performance_metrics : Dict[str, Any], optional
            성능 메트릭
            
        Returns:
        --------
        Dict[str, Any]
            완성된 메타데이터
        """
        metadata = {
            'model_info': model_info,
            'feature_extraction': {
                'sample_rate': self.config.sample_rate,
                'hop_length': self.config.hop_length,
                'n_mfcc': self.config.n_mfcc,
                'n_chroma': self.config.n_chroma,
                'feature_vector_size': 30
            },
            'class_information': {
                'class_names': self.config.class_names,
                'n_classes': len(self.config.class_names)
            },
            'created_at': datetime.now().isoformat(),
            'version': getattr(self.config, 'model_version', '1.0')
        }
        
        if training_info:
            metadata['training_info'] = training_info
        
        if performance_metrics:
            metadata['performance_metrics'] = performance_metrics
        
        return metadata
    
    def get_conversion_summary(self) -> Dict[str, Any]:
        """
        변환 작업 요약을 반환합니다.
        
        Returns:
        --------
        Dict[str, Any]
            변환 요약 정보
        """
        if not self.conversion_results:
            return {'total_conversions': 0}
        
        total_conversions = len(self.conversion_results)
        successful_conversions = sum(1 for r in self.conversion_results.values() if r.validation_passed)
        total_time = sum(r.conversion_time for r in self.conversion_results.values())
        total_size = sum(r.file_size_bytes for r in self.conversion_results.values())
        
        summary = {
            'total_conversions': total_conversions,
            'successful_conversions': successful_conversions,
            'success_rate': successful_conversions / total_conversions if total_conversions > 0 else 0,
            'total_conversion_time': total_time,
            'average_conversion_time': total_time / total_conversions if total_conversions > 0 else 0,
            'total_file_size_bytes': total_size,
            'average_file_size_bytes': total_size / total_conversions if total_conversions > 0 else 0,
            'coreml_available': COREML_AVAILABLE
        }
        
        return summary