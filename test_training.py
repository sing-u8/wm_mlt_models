#!/usr/bin/env python3
"""
머신러닝 훈련 모듈 테스트 스크립트
ModelTrainer 클래스 및 교차 검증 기능 검증
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import joblib
import json
sys.path.append('.')

from src.ml.training import ModelTrainer, ModelConfig, TrainingResult, ModelArtifact
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def create_test_dataset():
    """테스트용 가상 데이터셋 생성"""
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
    
    # train/validation/test 분할
    train_size = int(0.7 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def test_model_trainer_initialization():
    """ModelTrainer 초기화 테스트"""
    
    logger = setup_logger("trainer_init_test", "INFO")
    logger.info("=== ModelTrainer 초기화 테스트 시작 ===")
    
    try:
        # 기본 구성으로 초기화
        trainer = ModelTrainer()
        
        # 모델 구성 확인
        expected_models = ["svm", "random_forest"]
        actual_models = list(trainer.models.keys())
        
        if set(expected_models) == set(actual_models):
            logger.info(f"✅ 모델 구성 올바름: {actual_models}")
        else:
            logger.error(f"❌ 모델 구성 오류: 예상 {expected_models}, 실제 {actual_models}")
            return False
        
        # SVM 구성 확인
        svm_config = trainer.models["svm"]
        if svm_config.model_type == "svm" and "C" in svm_config.param_grid:
            logger.info("✅ SVM 구성 올바름")
        else:
            logger.error("❌ SVM 구성 오류")
            return False
        
        # Random Forest 구성 확인
        rf_config = trainer.models["random_forest"]
        if rf_config.model_type == "random_forest" and "n_estimators" in rf_config.param_grid:
            logger.info("✅ Random Forest 구성 올바름")
        else:
            logger.error("❌ Random Forest 구성 오류")
            return False
        
        logger.info("✅ ModelTrainer 초기화 성공")
        return True
        
    except Exception as e:
        logger.error(f"ModelTrainer 초기화 실패: {e}")
        return False
    
    finally:
        logger.info("=== ModelTrainer 초기화 테스트 완료 ===\n")

def test_single_model_training():
    """단일 모델 훈련 테스트"""
    
    logger = setup_logger("single_model_test", "INFO")
    logger.info("=== 단일 모델 훈련 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        logger.info(f"테스트 데이터 생성: {X_train.shape} 훈련 샘플")
        
        # ModelTrainer 초기화
        trainer = ModelTrainer()
        
        # SVM 모델 훈련
        logger.info("SVM 모델 훈련 시작...")
        svm_result = trainer.train_single_model("svm", X_train, y_train, cv_folds=3)
        
        # 결과 검증
        if isinstance(svm_result, TrainingResult):
            logger.info(f"✅ SVM 훈련 완료:")
            logger.info(f"  최적 점수: {svm_result.best_score:.4f}")
            logger.info(f"  최적 파라미터: {svm_result.best_params}")
            logger.info(f"  훈련 시간: {svm_result.training_time:.2f}초")
        else:
            logger.error("❌ SVM 훈련 결과 형식 오류")
            return False
        
        # Random Forest 모델 훈련 (더 작은 그리드로)
        logger.info("Random Forest 모델 훈련 시작...")
        
        # 빠른 테스트를 위해 하이퍼파라미터 그리드 축소
        original_rf_grid = trainer.models["random_forest"].param_grid
        trainer.models["random_forest"].param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
        
        rf_result = trainer.train_single_model("random_forest", X_train, y_train, cv_folds=3)
        
        # 원래 그리드 복원
        trainer.models["random_forest"].param_grid = original_rf_grid
        
        if isinstance(rf_result, TrainingResult):
            logger.info(f"✅ Random Forest 훈련 완료:")
            logger.info(f"  최적 점수: {rf_result.best_score:.4f}")
            logger.info(f"  최적 파라미터: {rf_result.best_params}")
            logger.info(f"  훈련 시간: {rf_result.training_time:.2f}초")
            logger.info(f"  특징 중요도 개수: {len(rf_result.feature_importance) if rf_result.feature_importance else 0}")
        else:
            logger.error("❌ Random Forest 훈련 결과 형식 오류")
            return False
        
        logger.info("✅ 단일 모델 훈련 성공")
        return True
        
    except Exception as e:
        logger.error(f"단일 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 단일 모델 훈련 테스트 완료 ===\n")

def test_cross_validation_training():
    """교차 검증 훈련 테스트"""
    
    logger = setup_logger("cv_training_test", "INFO")
    logger.info("=== 교차 검증 훈련 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        
        # ModelTrainer 초기화
        trainer = ModelTrainer()
        
        # 빠른 테스트를 위해 하이퍼파라미터 그리드 축소
        trainer.models["svm"].param_grid = {
            "C": [1, 10],
            "gamma": ["scale", "auto"]
        }
        trainer.models["random_forest"].param_grid = {
            "n_estimators": [50],
            "max_depth": [None, 10],
            "min_samples_split": [2]
        }
        
        # 모든 모델 교차 검증 훈련
        logger.info("모든 모델 교차 검증 훈련 시작...")
        all_results = trainer.train_with_cv(X_train, y_train, cv_folds=3)
        
        if len(all_results) != 2:
            logger.error(f"❌ 예상 모델 수 2개, 실제 {len(all_results)}개")
            return False
        
        # 결과 검증
        for model_type, result in all_results.items():
            if not isinstance(result, TrainingResult):
                logger.error(f"❌ {model_type} 결과 형식 오류")
                return False
            
            logger.info(f"✅ {model_type} 훈련 완료:")
            logger.info(f"  최적 점수: {result.best_score:.4f}")
            logger.info(f"  CV 점수 개수: {len(result.cv_scores)}")
            logger.info(f"  특징 개수: {result.n_features}")
            logger.info(f"  샘플 개수: {result.n_samples}")
        
        # 훈련된 모델이 저장되었는지 확인
        if len(trainer.trained_models) == 2:
            logger.info("✅ 모든 모델이 내부에 저장됨")
        else:
            logger.error(f"❌ 저장된 모델 수 오류: {len(trainer.trained_models)}")
            return False
        
        logger.info("✅ 교차 검증 훈련 성공")
        return True
        
    except Exception as e:
        logger.error(f"교차 검증 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 교차 검증 훈련 테스트 완료 ===\n")

def test_model_evaluation():
    """모델 평가 테스트"""
    
    logger = setup_logger("evaluation_test", "INFO")
    logger.info("=== 모델 평가 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        
        # ModelTrainer 초기화 및 훈련
        trainer = ModelTrainer()
        
        # 빠른 테스트를 위해 그리드 축소
        trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
        trainer.models["random_forest"].param_grid = {"n_estimators": [50], "max_depth": [10]}
        
        # 모델 훈련
        all_results = trainer.train_with_cv(X_train, y_train, cv_folds=3)
        
        # 개별 모델 평가
        logger.info("개별 모델 평가 시작...")
        svm_metrics = trainer.evaluate_single_model("svm", X_test, y_test)
        
        # 평가 메트릭 확인
        expected_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in expected_metrics:
            if metric not in svm_metrics:
                logger.error(f"❌ SVM 평가에서 {metric} 메트릭 누락")
                return False
        
        logger.info(f"✅ SVM 평가 완료: 정확도 {svm_metrics['accuracy']:.4f}")
        
        # 전체 모델 평가
        logger.info("전체 모델 평가 시작...")
        all_metrics = trainer.evaluate_final(X_test, y_test)
        
        if len(all_metrics) != 2:
            logger.error(f"❌ 평가된 모델 수 오류: {len(all_metrics)}")
            return False
        
        for model_type, metrics in all_metrics.items():
            logger.info(f"✅ {model_type} 최종 평가:")
            logger.info(f"  정확도: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        
        logger.info("✅ 모델 평가 성공")
        return True
        
    except Exception as e:
        logger.error(f"모델 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 모델 평가 테스트 완료 ===\n")

def test_model_saving_and_loading():
    """모델 저장 및 로딩 테스트"""
    
    logger = setup_logger("save_load_test", "INFO")
    logger.info("=== 모델 저장 및 로딩 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        
        # ModelTrainer 초기화 및 훈련
        trainer = ModelTrainer()
        
        # 빠른 테스트를 위해 그리드 축소
        trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
        trainer.models["random_forest"].param_grid = {"n_estimators": [50]}
        
        # 모델 훈련
        all_results = trainer.train_with_cv(X_train, y_train, cv_folds=3)
        
        # 임시 디렉토리에 모델 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 모델 저장: {temp_dir}")
            
            # 모델 저장
            saved_paths = trainer.save_models(temp_dir)
            
            if len(saved_paths) != 2:
                logger.error(f"❌ 저장된 모델 수 오류: {len(saved_paths)}")
                return False
            
            # 저장된 파일 확인
            for model_type, path in saved_paths.items():
                if not os.path.exists(path):
                    logger.error(f"❌ {model_type} 모델 파일이 존재하지 않음: {path}")
                    return False
                
                # 메타데이터 파일 확인
                config_path = os.path.join(temp_dir, f"{model_type}_config.json")
                if not os.path.exists(config_path):
                    logger.error(f"❌ {model_type} 메타데이터 파일이 존재하지 않음")
                    return False
                
                logger.info(f"✅ {model_type} 모델 및 메타데이터 저장 완료")
            
            # 모델 로딩 테스트
            for model_type, path in saved_paths.items():
                loaded_model = trainer.load_model(path)
                
                # 로딩된 모델로 예측 테스트
                predictions = loaded_model.predict(X_test[:5])  # 처음 5개 샘플만
                
                if len(predictions) == 5:
                    logger.info(f"✅ {model_type} 모델 로딩 및 예측 성공")
                else:
                    logger.error(f"❌ {model_type} 모델 예측 실패")
                    return False
            
            # 모델 아티팩트 확인
            for model_type in saved_paths.keys():
                if model_type in trainer.model_artifacts:
                    artifact = trainer.model_artifacts[model_type]
                    if isinstance(artifact, ModelArtifact):
                        logger.info(f"✅ {model_type} 아티팩트 생성 완료")
                    else:
                        logger.error(f"❌ {model_type} 아티팩트 형식 오류")
                        return False
        
        logger.info("✅ 모델 저장 및 로딩 성공")
        return True
        
    except Exception as e:
        logger.error(f"모델 저장/로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 모델 저장 및 로딩 테스트 완료 ===\n")

def test_feature_importance():
    """특징 중요도 테스트"""
    
    logger = setup_logger("feature_importance_test", "INFO")
    logger.info("=== 특징 중요도 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        
        # ModelTrainer 초기화 및 훈련
        trainer = ModelTrainer()
        
        # Random Forest만 훈련 (특징 중요도 지원)
        trainer.models["random_forest"].param_grid = {"n_estimators": [50]}
        
        rf_result = trainer.train_single_model("random_forest", X_train, y_train, cv_folds=3)
        
        # 특징 중요도 추출
        importance_pairs = trainer.get_feature_importance("random_forest")
        
        if importance_pairs is None:
            logger.error("❌ Random Forest 특징 중요도 추출 실패")
            return False
        
        if len(importance_pairs) != 30:  # 30차원 특징
            logger.error(f"❌ 특징 중요도 개수 오류: 예상 30, 실제 {len(importance_pairs)}")
            return False
        
        # 중요도 순으로 정렬되었는지 확인
        importances = [pair[1] for pair in importance_pairs]
        if importances != sorted(importances, reverse=True):
            logger.error("❌ 특징 중요도가 내림차순으로 정렬되지 않음")
            return False
        
        logger.info(f"✅ 특징 중요도 추출 성공: {len(importance_pairs)}개 특징")
        logger.info(f"  상위 3개 특징: {importance_pairs[:3]}")
        
        # SVM은 특징 중요도가 없어야 함
        svm_importance = trainer.get_feature_importance("svm")
        if svm_importance is not None:
            logger.error("❌ SVM에서 특징 중요도가 반환됨 (예상: None)")
            return False
        
        logger.info("✅ SVM 특징 중요도 올바르게 None 반환")
        
        logger.info("✅ 특징 중요도 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"특징 중요도 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 특징 중요도 테스트 완료 ===\n")

def test_training_summary():
    """훈련 요약 정보 테스트"""
    
    logger = setup_logger("summary_test", "INFO")
    logger.info("=== 훈련 요약 정보 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        X_train, y_train, X_val, y_val, X_test, y_test = create_test_dataset()
        
        # ModelTrainer 초기화 및 훈련
        trainer = ModelTrainer()
        
        # 빠른 테스트를 위해 그리드 축소
        trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
        trainer.models["random_forest"].param_grid = {"n_estimators": [50]}
        
        # 모델 훈련
        all_results = trainer.train_with_cv(X_train, y_train, cv_folds=3)
        
        # 훈련 요약 정보 생성
        summary = trainer.get_training_summary()
        
        # 요약 정보 검증
        if summary['total_models'] != 2:
            logger.error(f"❌ 총 모델 수 오류: {summary['total_models']}")
            return False
        
        expected_models = {'svm', 'random_forest'}
        actual_models = set(summary['trained_models'])
        if expected_models != actual_models:
            logger.error(f"❌ 훈련된 모델 목록 오류: {actual_models}")
            return False
        
        # 각 모델의 요약 정보 확인
        for model_type in expected_models:
            if model_type not in summary['training_results']:
                logger.error(f"❌ {model_type} 훈련 결과 누락")
                return False
            
            result_summary = summary['training_results'][model_type]
            required_keys = ['best_score', 'best_params', 'training_time', 'n_features', 'n_samples']
            
            for key in required_keys:
                if key not in result_summary:
                    logger.error(f"❌ {model_type} 요약에서 {key} 누락")
                    return False
        
        logger.info("✅ 훈련 요약 정보:")
        logger.info(f"  총 모델 수: {summary['total_models']}")
        logger.info(f"  훈련된 모델: {summary['trained_models']}")
        
        for model_type, result in summary['training_results'].items():
            logger.info(f"  {model_type}: 점수 {result['best_score']:.4f}, "
                       f"특징 {result['n_features']}개, 샘플 {result['n_samples']}개")
        
        logger.info("✅ 훈련 요약 정보 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"훈련 요약 정보 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 훈련 요약 정보 테스트 완료 ===\n")

if __name__ == "__main__":
    logger = setup_logger("main_training_test", "INFO")
    logger.info("🤖 머신러닝 훈련 모듈 종합 테스트 시작 🤖")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("ModelTrainer 초기화", test_model_trainer_initialization()))
    test_results.append(("단일 모델 훈련", test_single_model_training()))
    test_results.append(("교차 검증 훈련", test_cross_validation_training()))
    test_results.append(("모델 평가", test_model_evaluation()))
    test_results.append(("모델 저장/로딩", test_model_saving_and_loading()))
    test_results.append(("특징 중요도", test_feature_importance()))
    test_results.append(("훈련 요약 정보", test_training_summary()))
    
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
        logger.info("✅ ModelTrainer 클래스가 올바르게 구현되었습니다.")
        logger.info("✅ 교차 검증 및 하이퍼파라미터 최적화가 제대로 작동합니다.")
        logger.info("✅ 모델 저장/로딩 기능이 정상 작동합니다.")
    else:
        logger.info("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("🤖 머신러닝 훈련 모듈 테스트 완료 🤖")