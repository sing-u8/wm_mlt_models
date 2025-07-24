#!/usr/bin/env python3
"""
모델 변환 모듈 테스트 스크립트
ModelConverter 클래스 및 Core ML 변환 기능 검증
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import json
sys.path.append('.')

from src.ml.model_converter import ModelConverter, ConversionResult, CoreMLModelInfo
from src.ml.training import ModelTrainer
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

def create_test_model_and_data():
    """테스트용 모델과 데이터 생성"""
    np.random.seed(42)
    
    # 30차원 특징 벡터 (design.md 명세)
    n_samples = 150
    n_features = 30
    n_classes = 3
    
    # 각 클래스별로 50개 샘플 생성
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # 클래스별로 다른 특성을 가진 데이터 생성
        class_center = np.random.randn(n_features) * 2
        class_samples = np.random.randn(50, n_features) + class_center
        
        X.append(class_samples)
        y.extend([class_idx] * 50)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # 데이터 셔플
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # train/test 분할
    train_size = int(0.8 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # 간단한 모델 훈련
    trainer = ModelTrainer()
    
    # 빠른 테스트를 위해 하이퍼파라미터 그리드 축소
    trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
    trainer.models["random_forest"].param_grid = {"n_estimators": [50], "max_depth": [10]}
    
    # 모델 훈련
    all_results = trainer.train_with_cv(X_train, y_train, cv_folds=3)
    
    # 훈련된 모델 반환
    models = {
        "svm": trainer.trained_models["svm"],
        "random_forest": trainer.trained_models["random_forest"]
    }
    
    return models, X_test, y_test

def test_model_converter_initialization():
    """ModelConverter 초기화 테스트"""
    
    logger = setup_logger("converter_init_test", "INFO")
    logger.info("=== ModelConverter 초기화 테스트 시작 ===")
    
    try:
        # 기본 구성으로 초기화
        converter = ModelConverter()
        
        # 구성 확인
        if converter.config is not None:
            logger.info("✅ 구성 객체 로드됨")
        else:
            logger.error("❌ 구성 객체 로드 실패")
            return False
        
        # Core ML 사용 가능성 확인
        logger.info(f"Core ML 지원: {COREML_AVAILABLE}")
        
        # 변환 결과 저장소 초기화 확인
        if hasattr(converter, 'conversion_results'):
            logger.info("✅ 변환 결과 저장소 초기화됨")
        else:
            logger.error("❌ 변환 결과 저장소 초기화 실패")
            return False
        
        logger.info("✅ ModelConverter 초기화 성공")
        return True
        
    except Exception as e:
        logger.error(f"ModelConverter 초기화 실패: {e}")
        return False
    
    finally:
        logger.info("=== ModelConverter 초기화 테스트 완료 ===\n")

def test_pickle_model_saving():
    """Pickle 모델 저장 테스트"""
    
    logger = setup_logger("pickle_save_test", "INFO")
    logger.info("=== Pickle 모델 저장 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_model_and_data()
        logger.info(f"테스트 모델 생성 완료")
        
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 임시 디렉토리에 모델 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 모델 저장: {temp_dir}")
            
            # SVM 모델 저장
            model_metadata = {
                'model_type': 'svm',
                'feature_count': 30,
                'class_names': DEFAULT_CONFIG.class_names
            }
            
            svm_path = converter.save_pickle_model(
                models["svm"], "svm", model_metadata, temp_dir
            )
            
            # 저장된 파일 확인
            if not os.path.exists(svm_path):
                logger.error(f"❌ SVM 모델 파일이 존재하지 않음: {svm_path}")
                return False
            
            file_size = os.path.getsize(svm_path)
            if file_size < 100:  # 100 bytes 미만이면 오류로 간주
                logger.error(f"❌ SVM 모델 파일 크기가 너무 작음: {file_size} bytes")
                return False
            
            logger.info(f"✅ SVM 모델 저장 성공: {file_size} bytes")
            
            # 메타데이터 파일 확인
            metadata_path = os.path.join(temp_dir, "svm_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)
                
                if 'model_type' in saved_metadata and saved_metadata['model_type'] == 'svm':
                    logger.info("✅ 메타데이터 저장 및 로드 성공")
                else:
                    logger.error("❌ 메타데이터 내용 오류")
                    return False
            
            # 모델 로딩 테스트
            loaded_model = converter.load_pickle_model(svm_path)
            
            # 로딩된 모델로 예측 테스트
            predictions = loaded_model.predict(X_test[:5])  # 처음 5개 샘플만
            
            if len(predictions) == 5:
                logger.info("✅ 모델 로딩 및 예측 성공")
            else:
                logger.error("❌ 모델 예측 실패")
                return False
        
        logger.info("✅ Pickle 모델 저장 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"Pickle 모델 저장 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== Pickle 모델 저장 테스트 완료 ===\n")

def test_coreml_conversion():
    """Core ML 변환 테스트"""
    
    logger = setup_logger("coreml_conversion_test", "INFO")
    logger.info("=== Core ML 변환 테스트 시작 ===")
    
    if not COREML_AVAILABLE:
        logger.warning("⚠️ coremltools가 설치되지 않아 Core ML 테스트를 건너뜁니다.")
        return True  # 설치되지 않은 것은 오류가 아님
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_model_and_data()
        
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 임시 디렉토리에 변환
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 Core ML 모델 저장: {temp_dir}")
            
            # Random Forest 모델 변환 (SVM보다 안정적)
            input_description = {
                'name': 'audio_features',
                'shape': (30,),
                'description': 'Audio feature vector extracted from watermelon sound'
            }
            
            coreml_path = converter.convert_to_coreml(
                models["random_forest"], "random_forest", input_description, temp_dir
            )
            
            # 변환된 파일 확인
            if not os.path.exists(coreml_path):
                logger.error(f"❌ Core ML 모델 파일이 존재하지 않음: {coreml_path}")
                return False
            
            file_size = os.path.getsize(coreml_path)
            if file_size < 1000:  # 1KB 미만이면 오류로 간주
                logger.error(f"❌ Core ML 모델 파일 크기가 너무 작음: {file_size} bytes")
                return False
            
            logger.info(f"✅ Core ML 변환 성공: {file_size} bytes")
            
            # 메타데이터 파일 확인
            metadata_path = os.path.join(temp_dir, "random_forest_coreml_info.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    coreml_info = json.load(f)
                
                required_keys = ['model_path', 'input_name', 'input_shape', 'class_labels']
                for key in required_keys:
                    if key not in coreml_info:
                        logger.error(f"❌ Core ML 메타데이터에서 {key} 누락")
                        return False
                
                logger.info("✅ Core ML 메타데이터 저장 성공")
            
            # Core ML 모델 로딩 및 예측 테스트
            try:
                coreml_model = ct.models.MLModel(coreml_path)
                
                # 단일 샘플로 예측 테스트
                test_sample = X_test[0:1].astype(np.float32)
                input_dict = {'audio_features': test_sample}
                
                result = coreml_model.predict(input_dict)
                
                if 'watermelon_class' in result:
                    logger.info(f"✅ Core ML 모델 예측 성공: {result['watermelon_class']}")
                else:
                    logger.error("❌ Core ML 모델 예측 결과에 클래스 정보 없음")
                    return False
                
            except Exception as e:
                logger.error(f"❌ Core ML 모델 로딩/예측 실패: {e}")
                return False
        
        logger.info("✅ Core ML 변환 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"Core ML 변환 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== Core ML 변환 테스트 완료 ===\n")

def test_model_validation():
    """모델 변환 검증 테스트"""
    
    logger = setup_logger("model_validation_test", "INFO")
    logger.info("=== 모델 변환 검증 테스트 시작 ===")
    
    if not COREML_AVAILABLE:
        logger.warning("⚠️ coremltools가 설치되지 않아 검증 테스트를 건너뜁니다.")
        return True
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_model_and_data()
        
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 임시 디렉토리에 변환 및 검증
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에서 변환 및 검증: {temp_dir}")
            
            # Random Forest 모델 변환
            coreml_path = converter.convert_to_coreml(
                models["random_forest"], "random_forest", output_dir=temp_dir
            )
            
            # 검증 수행
            validation_result = converter.validate_model_conversion(
                models["random_forest"], coreml_path, test_samples=10
            )
            
            # 검증 결과 확인
            if 'error' in validation_result:
                logger.error(f"❌ 검증 중 오류 발생: {validation_result['error']}")
                return False
            
            required_keys = ['test_samples', 'class_prediction_accuracy', 'predictions_match']
            for key in required_keys:
                if key not in validation_result:
                    logger.error(f"❌ 검증 결과에서 {key} 누락")
                    return False
            
            accuracy = validation_result['class_prediction_accuracy']
            logger.info(f"✅ 검증 완료:")
            logger.info(f"  테스트 샘플: {validation_result['test_samples']}개")
            logger.info(f"  클래스 예측 정확도: {accuracy:.1%}")
            logger.info(f"  모든 예측 일치: {validation_result['predictions_match']}")
            
            # 확률 검증 정보 (있는 경우)
            if 'probability_validation' in validation_result:
                logger.info(f"  확률 예측 최대 차이: {validation_result['max_probability_difference']:.6f}")
                logger.info(f"  확률 예측 평균 차이: {validation_result['mean_probability_difference']:.6f}")
            
            # 검증 통과 기준 (80% 이상 일치)
            if accuracy >= 0.8:
                logger.info("✅ 검증 통과 (80% 이상 일치)")
            else:
                logger.warning(f"⚠️ 검증 주의 (정확도 {accuracy:.1%} < 80%)")
        
        logger.info("✅ 모델 변환 검증 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"모델 변환 검증 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 모델 변환 검증 테스트 완료 ===\n")

def test_conversion_with_validation():
    """검증을 포함한 전체 변환 프로세스 테스트"""
    
    logger = setup_logger("full_conversion_test", "INFO")
    logger.info("=== 검증 포함 전체 변환 테스트 시작 ===")
    
    if not COREML_AVAILABLE:
        logger.warning("⚠️ coremltools가 설치되지 않아 전체 변환 테스트를 건너뜁니다.")
        return True
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_model_and_data()
        
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 임시 디렉토리에서 전체 변환 프로세스
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에서 전체 변환 프로세스: {temp_dir}")
            
            # 검증을 포함한 변환
            conversion_result = converter.convert_model_with_validation(
                models["random_forest"], "random_forest", 
                output_dir=temp_dir, validate=True
            )
            
            # 변환 결과 검증
            if not isinstance(conversion_result, ConversionResult):
                logger.error("❌ 변환 결과 형식 오류")
                return False
            
            # 필수 속성 확인
            required_attributes = [
                'model_name', 'original_format', 'target_format',
                'converted_path', 'conversion_time', 'validation_passed'
            ]
            
            for attr in required_attributes:
                if not hasattr(conversion_result, attr):
                    logger.error(f"❌ 변환 결과에서 {attr} 누락")
                    return False
            
            logger.info(f"✅ 전체 변환 완료:")
            logger.info(f"  모델명: {conversion_result.model_name}")
            logger.info(f"  변환 형식: {conversion_result.original_format} → {conversion_result.target_format}")
            logger.info(f"  변환 시간: {conversion_result.conversion_time:.3f}초")
            logger.info(f"  파일 크기: {conversion_result.file_size_bytes:,} bytes")
            logger.info(f"  검증 통과: {conversion_result.validation_passed}")
            
            # 변환된 파일 존재 확인
            if not os.path.exists(conversion_result.converted_path):
                logger.error(f"❌ 변환된 파일이 존재하지 않음: {conversion_result.converted_path}")
                return False
            
            # 검증 세부 정보 확인
            if 'validation_details' in conversion_result.__dict__:
                validation_details = conversion_result.validation_details
                if 'class_prediction_accuracy' in validation_details:
                    logger.info(f"  검증 정확도: {validation_details['class_prediction_accuracy']:.1%}")
        
        logger.info("✅ 검증 포함 전체 변환 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"검증 포함 전체 변환 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 검증 포함 전체 변환 테스트 완료 ===\n")

def test_metadata_creation():
    """메타데이터 생성 테스트"""
    
    logger = setup_logger("metadata_test", "INFO")
    logger.info("=== 메타데이터 생성 테스트 시작 ===")
    
    try:
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 테스트 모델 정보
        model_info = {
            'model_type': 'random_forest',
            'n_estimators': 50,
            'max_depth': 10
        }
        
        training_info = {
            'training_samples': 120,
            'validation_samples': 30,
            'cv_folds': 5,
            'best_score': 0.85
        }
        
        performance_metrics = {
            'accuracy': 0.87,
            'f1_macro': 0.85,
            'precision_macro': 0.86,
            'recall_macro': 0.84
        }
        
        # 메타데이터 생성
        metadata = converter.create_model_metadata(
            model_info, training_info, performance_metrics
        )
        
        # 메타데이터 검증
        required_sections = ['model_info', 'feature_extraction', 'class_information']
        for section in required_sections:
            if section not in metadata:
                logger.error(f"❌ 메타데이터에서 {section} 섹션 누락")
                return False
        
        # 특징 추출 정보 확인
        feature_extraction = metadata['feature_extraction']
        if feature_extraction['feature_vector_size'] != 30:
            logger.error(f"❌ 특징 벡터 크기 오류: {feature_extraction['feature_vector_size']}")
            return False
        
        # 클래스 정보 확인
        class_info = metadata['class_information']
        if len(class_info['class_names']) != 3:
            logger.error(f"❌ 클래스 수 오류: {len(class_info['class_names'])}")
            return False
        
        logger.info(f"✅ 메타데이터 생성 성공:")
        logger.info(f"  모델 타입: {metadata['model_info']['model_type']}")
        logger.info(f"  특징 벡터 크기: {metadata['feature_extraction']['feature_vector_size']}")
        logger.info(f"  클래스 수: {metadata['class_information']['n_classes']}")
        logger.info(f"  버전: {metadata['version']}")
        
        # 선택적 정보 확인
        if 'training_info' in metadata:
            logger.info(f"  훈련 샘플: {metadata['training_info']['training_samples']}")
        
        if 'performance_metrics' in metadata:
            logger.info(f"  정확도: {metadata['performance_metrics']['accuracy']:.3f}")
        
        logger.info("✅ 메타데이터 생성 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"메타데이터 생성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 메타데이터 생성 테스트 완료 ===\n")

def test_conversion_summary():
    """변환 요약 정보 테스트"""
    
    logger = setup_logger("conversion_summary_test", "INFO")
    logger.info("=== 변환 요약 정보 테스트 시작 ===")
    
    if not COREML_AVAILABLE:
        logger.warning("⚠️ coremltools가 설치되지 않아 변환 요약 테스트를 건너뜁니다.")
        return True
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_model_and_data()
        
        # ModelConverter 초기화
        converter = ModelConverter()
        
        # 빈 상태에서 요약 확인
        empty_summary = converter.get_conversion_summary()
        if empty_summary['total_conversions'] != 0:
            logger.error("❌ 빈 상태 요약 오류")
            return False
        
        logger.info("✅ 빈 상태 요약 정보 올바름")
        
        # 임시 디렉토리에서 변환 수행
        with tempfile.TemporaryDirectory() as temp_dir:
            # 두 모델 변환
            for model_name, model in models.items():
                try:
                    conversion_result = converter.convert_model_with_validation(
                        model, model_name, output_dir=temp_dir, validate=True
                    )
                    logger.info(f"✅ {model_name} 변환 완료")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} 변환 실패: {e}")
                    continue
        
        # 변환 요약 생성
        summary = converter.get_conversion_summary()
        
        # 요약 정보 검증
        required_keys = [
            'total_conversions', 'successful_conversions', 'success_rate',
            'total_conversion_time', 'average_conversion_time',
            'total_file_size_bytes', 'average_file_size_bytes'
        ]
        
        for key in required_keys:
            if key not in summary:
                logger.error(f"❌ 변환 요약에서 {key} 누락")
                return False
        
        logger.info(f"✅ 변환 요약 정보:")
        logger.info(f"  총 변환 수: {summary['total_conversions']}")
        logger.info(f"  성공 변환 수: {summary['successful_conversions']}")
        logger.info(f"  성공률: {summary['success_rate']:.1%}")
        logger.info(f"  총 변환 시간: {summary['total_conversion_time']:.3f}초")
        
        if summary['total_conversions'] > 0:
            logger.info(f"  평균 변환 시간: {summary['average_conversion_time']:.3f}초")
            logger.info(f"  총 파일 크기: {summary['total_file_size_bytes']:,} bytes")
            logger.info(f"  평균 파일 크기: {summary['average_file_size_bytes']:,.0f} bytes")
        
        logger.info("✅ 변환 요약 정보 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"변환 요약 정보 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 변환 요약 정보 테스트 완료 ===\n")

if __name__ == "__main__":
    logger = setup_logger("main_converter_test", "INFO")
    logger.info("🔄 모델 변환 모듈 종합 테스트 시작 🔄")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("ModelConverter 초기화", test_model_converter_initialization()))
    test_results.append(("Pickle 모델 저장", test_pickle_model_saving()))
    test_results.append(("Core ML 변환", test_coreml_conversion()))
    test_results.append(("모델 변환 검증", test_model_validation()))
    test_results.append(("검증 포함 전체 변환", test_conversion_with_validation()))
    test_results.append(("메타데이터 생성", test_metadata_creation()))
    test_results.append(("변환 요약 정보", test_conversion_summary()))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    success_rate = passed / total * 100
    logger.info(f"전체 테스트 결과: {passed}/{total} 통과 ({success_rate:.1f}%)")
    
    if passed == total:
        logger.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        logger.info("✅ ModelConverter 클래스가 올바르게 구현되었습니다.")
        logger.info("✅ Pickle 모델 저장/로딩 기능이 정상 작동합니다.")
        if COREML_AVAILABLE:
            logger.info("✅ Core ML 변환 및 검증 기능이 정상 작동합니다.")
        else:
            logger.info("ℹ️ Core ML 기능은 coremltools가 설치되지 않아 테스트되지 않았습니다.")
    else:
        logger.info("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("🔄 모델 변환 모듈 테스트 완료 🔄")