#!/usr/bin/env python3
"""
모델 평가 모듈 테스트 스크립트
ModelEvaluator 클래스 및 평가 메트릭 기능 검증
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 matplotlib 사용
import matplotlib.pyplot as plt
sys.path.append('.')

from src.ml.evaluation import (
    ModelEvaluator, ClassificationMetrics, ModelComparison, EvaluationReport
)
from src.ml.training import ModelTrainer
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def create_test_models_and_data():
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

def test_model_evaluator_initialization():
    """ModelEvaluator 초기화 테스트"""
    
    logger = setup_logger("evaluator_init_test", "INFO")
    logger.info("=== ModelEvaluator 초기화 테스트 시작 ===")
    
    try:
        # 기본 구성으로 초기화
        evaluator = ModelEvaluator()
        
        # 구성 확인
        if evaluator.config is not None:
            logger.info("✅ 구성 객체 로드됨")
        else:
            logger.error("❌ 구성 객체 로드 실패")
            return False
        
        # 결과 저장소 초기화 확인
        if hasattr(evaluator, 'evaluation_results'):
            logger.info("✅ 평가 결과 저장소 초기화됨")
        else:
            logger.error("❌ 평가 결과 저장소 초기화 실패")
            return False
        
        logger.info("✅ ModelEvaluator 초기화 성공")
        return True
        
    except Exception as e:
        logger.error(f"ModelEvaluator 초기화 실패: {e}")
        return False
    
    finally:
        logger.info("=== ModelEvaluator 초기화 테스트 완료 ===\n")

def test_single_model_evaluation():
    """단일 모델 평가 테스트"""
    
    logger = setup_logger("single_eval_test", "INFO")
    logger.info("=== 단일 모델 평가 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        logger.info(f"테스트 데이터: {X_test.shape} 테스트 샘플")
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # SVM 모델 평가
        logger.info("SVM 모델 평가 시작...")
        svm_metrics = evaluator.evaluate_model(
            models["svm"], X_test, y_test, "svm"
        )
        
        # 결과 검증
        if not isinstance(svm_metrics, ClassificationMetrics):
            logger.error("❌ SVM 평가 결과 형식 오류")
            return False
        
        # 필수 메트릭 확인
        required_attributes = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'class_precision', 'class_recall', 'class_f1', 'confusion_matrix'
        ]
        
        for attr in required_attributes:
            if not hasattr(svm_metrics, attr):
                logger.error(f"❌ SVM 평가 결과에서 {attr} 누락")
                return False
        
        logger.info(f"✅ SVM 평가 완료:")
        logger.info(f"  정확도: {svm_metrics.accuracy:.4f}")
        logger.info(f"  F1 (macro): {svm_metrics.f1_macro:.4f}")
        logger.info(f"  혼동 행렬 형태: {svm_metrics.confusion_matrix.shape}")
        
        # 메트릭 값 범위 검증
        if not (0 <= svm_metrics.accuracy <= 1):
            logger.error(f"❌ 정확도 값 범위 오류: {svm_metrics.accuracy}")
            return False
        
        if not (0 <= svm_metrics.f1_macro <= 1):
            logger.error(f"❌ F1-score 값 범위 오류: {svm_metrics.f1_macro}")
            return False
        
        # 클래스별 메트릭 확인
        expected_classes = set(DEFAULT_CONFIG.class_names)
        actual_classes = set(svm_metrics.class_precision.keys())
        if expected_classes != actual_classes:
            logger.error(f"❌ 클래스별 메트릭 클래스 오류: {actual_classes}")
            return False
        
        logger.info("✅ 단일 모델 평가 성공")
        return True
        
    except Exception as e:
        logger.error(f"단일 모델 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 단일 모델 평가 테스트 완료 ===\n")

def test_model_comparison():
    """모델 비교 테스트"""
    
    logger = setup_logger("model_comparison_test", "INFO")
    logger.info("=== 모델 비교 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 모델 비교
        logger.info("SVM vs Random Forest 비교 시작...")
        comparison = evaluator.compare_models(
            models["svm"], models["random_forest"],
            X_test, y_test,
            "svm", "random_forest",
            cv_folds=3  # 빠른 테스트를 위해 3-fold
        )
        
        # 결과 검증
        if not isinstance(comparison, ModelComparison):
            logger.error("❌ 모델 비교 결과 형식 오류")
            return False
        
        # 필수 속성 확인
        required_attributes = [
            'model1_name', 'model2_name', 'model1_metrics', 'model2_metrics',
            'accuracy_diff', 'f1_macro_diff', 'better_model'
        ]
        
        for attr in required_attributes:
            if not hasattr(comparison, attr):
                logger.error(f"❌ 모델 비교 결과에서 {attr} 누락")
                return False
        
        logger.info(f"✅ 모델 비교 완료:")
        logger.info(f"  모델 1: {comparison.model1_name}")
        logger.info(f"  모델 2: {comparison.model2_name}")
        logger.info(f"  정확도 차이: {comparison.accuracy_diff:+.4f}")
        logger.info(f"  F1 차이: {comparison.f1_macro_diff:+.4f}")
        logger.info(f"  더 좋은 모델: {comparison.better_model}")
        
        # 통계적 유의성 테스트 결과 확인
        if 'pvalue' in comparison.accuracy_ttest:
            logger.info(f"  정확도 t-test p-value: {comparison.accuracy_ttest['pvalue']:.4f}")
        
        # 더 좋은 모델이 올바르게 선정되었는지 확인
        if comparison.better_model not in [comparison.model1_name, comparison.model2_name]:
            logger.error(f"❌ 더 좋은 모델 선정 오류: {comparison.better_model}")
            return False
        
        logger.info("✅ 모델 비교 성공")
        return True
        
    except Exception as e:
        logger.error(f"모델 비교 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally: 
        logger.info("=== 모델 비교 테스트 완료 ===\n")

def test_confusion_matrix_visualization():
    """혼동 행렬 시각화 테스트"""
    
    logger = setup_logger("confusion_viz_test", "INFO")
    logger.info("=== 혼동 행렬 시각화 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 모델 평가
        svm_metrics = evaluator.evaluate_model(
            models["svm"], X_test, y_test, "svm"
        )
        
        # 임시 디렉토리에 시각화 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 시각화 저장: {temp_dir}")
            
            # 혼동 행렬 시각화
            save_path = os.path.join(temp_dir, "test_confusion_matrix.png")
            saved_path = evaluator.plot_confusion_matrix(
                svm_metrics, "svm", save_path=save_path
            )
            
            # 파일 생성 확인
            if not os.path.exists(saved_path):
                logger.error(f"❌ 혼동 행렬 시각화 파일이 생성되지 않음: {saved_path}")
                return False
            
            file_size = os.path.getsize(saved_path)
            if file_size < 1000:  # 1KB 미만이면 오류로 간주
                logger.error(f"❌ 혼동 행렬 시각화 파일 크기가 너무 작음: {file_size} bytes")
                return False
            
            logger.info(f"✅ 혼동 행렬 시각화 생성 완료: {file_size} bytes")
        
        logger.info("✅ 혼동 행렬 시각화 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"혼동 행렬 시각화 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 
        return False
    
    finally:
        logger.info("=== 혼동 행렬 시각화 테스트 완료 ===\n")

def test_model_comparison_visualization():
    """모델 비교 시각화 테스트"""
    
    logger = setup_logger("comparison_viz_test", "INFO")
    logger.info("=== 모델 비교 시각화 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 모델 비교
        comparison = evaluator.compare_models(
            models["svm"], models["random_forest"],
            X_test, y_test,
            "svm", "random_forest",
            cv_folds=3
        )
        
        # 임시 디렉토리에 시각화 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 시각화 저장: {temp_dir}")
            
            # 모델 비교 시각화
            save_path = os.path.join(temp_dir, "test_model_comparison.png")
            saved_path = evaluator.plot_model_comparison(comparison, save_path=save_path)
            
            # 파일 생성 확인
            if not os.path.exists(saved_path):
                logger.error(f"❌ 모델 비교 시각화 파일이 생성되지 않음: {saved_path}")
                return False
            
            file_size = os.path.getsize(saved_path)
            if file_size < 5000:  # 5KB 미만이면 오류로 간주 (더 복잡한 차트)
                logger.error(f"❌ 모델 비교 시각화 파일 크기가 너무 작음: {file_size} bytes")
                return False
            
            logger.info(f"✅ 모델 비교 시각화 생성 완료: {file_size} bytes")
        
        logger.info("✅ 모델 비교 시각화 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"모델 비교 시각화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 모델 비교 시각화 테스트 완료 ===\n")

def test_evaluation_report_generation():
    """종합 평가 보고서 생성 테스트"""
    
    logger = setup_logger("report_gen_test", "INFO")
    logger.info("=== 종합 평가 보고서 생성 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 데이터셋 정보
        dataset_info = {
            'test_samples': len(y_test),
            'n_features': X_test.shape[1],
            'n_classes': len(np.unique(y_test)),
            'description': 'Test dataset for evaluation'
        }
        
        # 종합 평가 보고서 생성
        logger.info("종합 평가 보고서 생성 시작...")
        report = evaluator.generate_evaluation_report(
            models, X_test, y_test, 
            dataset_info=dataset_info,
            save_report=False  # 메모리에만 보관
        )
        
        # 보고서 검증
        if not isinstance(report, EvaluationReport):
            logger.error("❌ 평가 보고서 형식 오류")
            return False
        
        # 필수 속성 확인
        required_attributes = [
            'evaluation_id', 'created_at', 'dataset_info', 'model_metrics'
        ]
        
        for attr in required_attributes:
            if not hasattr(report, attr):
                logger.error(f"❌ 평가 보고서에서 {attr} 누락")
                return False
        
        # 모델 메트릭 확인
        if len(report.model_metrics) != 2:
            logger.error(f"❌ 모델 메트릭 수 오류: {len(report.model_metrics)}")
            return False
        
        expected_models = {"svm", "random_forest"}
        actual_models = set(report.model_metrics.keys())
        if expected_models != actual_models:
            logger.error(f"❌ 모델 메트릭 모델명 오류: {actual_models}")
            return False
        
        # 모델 비교 확인
        if report.model_comparison is None:
            logger.error("❌ 모델 비교 결과가 없음")
            return False
        
        logger.info(f"✅ 종합 평가 보고서 생성 완료:")
        logger.info(f"  평가 ID: {report.evaluation_id}")
        logger.info(f"  모델 수: {len(report.model_metrics)}")
        logger.info(f"  데이터셋 샘플 수: {report.dataset_info['test_samples']}")
        logger.info(f"  모델 비교 포함: {report.model_comparison is not None}")
        
        # 보고서 요약 테스트
        summary = evaluator.get_evaluation_summary(report)
        
        if 'best_model' not in summary:
            logger.error("❌ 보고서 요약에 최고 성능 모델 정보 누락")
            return False
        
        logger.info(f"  최고 성능 모델: {summary['best_model']['name']} "
                   f"(F1: {summary['best_model']['f1_macro']:.4f})")
        
        logger.info("✅ 종합 평가 보고서 생성 성공")
        return True
        
    except Exception as e:
        logger.error(f"종합 평가 보고서 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 종합 평가 보고서 생성 테스트 완료 ===\n")

def test_report_save_and_load():
    """보고서 저장 및 로딩 테스트"""
    
    logger = setup_logger("report_save_load_test", "INFO")
    logger.info("=== 보고서 저장 및 로딩 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 종합 평가 보고서 생성
        report = evaluator.generate_evaluation_report(
            models, X_test, y_test, save_report=False
        )
        
        # 임시 디렉토리에 보고서 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"임시 디렉토리에 보고서 저장: {temp_dir}")
            
            # 임시 구성으로 저장 경로 변경
            original_output_dir = evaluator.config.model_output_dir
            evaluator.config.model_output_dir = temp_dir
            
            try:
                # 보고서 저장
                saved_path = evaluator.save_evaluation_report(report)
                
                # 파일 생성 확인
                if not os.path.exists(saved_path):
                    logger.error(f"❌ 보고서 파일이 생성되지 않음: {saved_path}")
                    return False
                
                file_size = os.path.getsize(saved_path)
                if file_size < 100:  # 100 bytes 미만이면 오류로 간주
                    logger.error(f"❌ 보고서 파일 크기가 너무 작음: {file_size} bytes")
                    return False
                
                logger.info(f"✅ 보고서 저장 완료: {file_size} bytes")
                
                # 보고서 로딩
                loaded_report = evaluator.load_evaluation_report(saved_path)
                
                # 로딩된 보고서 검증
                if loaded_report is None:
                    logger.error("❌ 보고서 로딩 실패")
                    return False
                
                # 기본 정보 비교
                if loaded_report['evaluation_id'] != report.evaluation_id:
                    logger.error("❌ 로딩된 보고서 ID 불일치")
                    return False
                
                logger.info(f"✅ 보고서 로딩 완료: {loaded_report['evaluation_id']}")
                
            finally:
                # 원래 구성 복원
                evaluator.config.model_output_dir = original_output_dir
        
        logger.info("✅ 보고서 저장 및 로딩 성공")
        return True
        
    except Exception as e:
        logger.error(f"보고서 저장/로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 보고서 저장 및 로딩 테스트 완료 ===\n")

def test_evaluation_summary():
    """평가 요약 정보 테스트"""
    
    logger = setup_logger("eval_summary_test", "INFO")
    logger.info("=== 평가 요약 정보 테스트 시작 ===")
    
    try:
        # 테스트 데이터 및 모델 생성
        models, X_test, y_test = create_test_models_and_data()
        
        # ModelEvaluator 초기화
        evaluator = ModelEvaluator()
        
        # 종합 평가 보고서 생성
        report = evaluator.generate_evaluation_report(
            models, X_test, y_test, save_report=False
        )
        
        # 평가 요약 생성
        summary = evaluator.get_evaluation_summary(report)
        
        # 요약 정보 검증
        required_keys = ['evaluation_id', 'total_models', 'model_performance', 'best_model']
        for key in required_keys:
            if key not in summary:
                logger.error(f"❌ 평가 요약에서 {key} 누락")
                return False
        
        # 모델별 성능 정보 확인
        if len(summary['model_performance']) != 2:
            logger.error(f"❌ 모델 성능 정보 수 오류: {len(summary['model_performance'])}")
            return False
        
        # 최고 성능 모델 정보 확인
        best_model = summary['best_model']
        if 'name' not in best_model or 'f1_macro' not in best_model:
            logger.error("❌ 최고 성능 모델 정보 불완전")
            return False
        
        logger.info(f"✅ 평가 요약 정보:")
        logger.info(f"  평가 ID: {summary['evaluation_id']}")
        logger.info(f"  총 모델 수: {summary['total_models']}")
        logger.info(f"  최고 성능 모델: {best_model['name']} (F1: {best_model['f1_macro']:.4f})")
        
        # 모델 비교 정보 확인
        if 'model_comparison' in summary:
            comparison = summary['model_comparison']
            logger.info(f"  모델 비교 결과: {comparison['better_model']} 승리")
            logger.info(f"  통계적 유의성: {comparison['statistically_significant']}")
        
        logger.info("✅ 평가 요약 정보 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"평가 요약 정보 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 평가 요약 정보 테스트 완료 ===\n")

if __name__ == "__main__":
    logger = setup_logger("main_evaluation_test", "INFO")
    logger.info("📊 모델 평가 모듈 종합 테스트 시작 📊")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("ModelEvaluator 초기화", test_model_evaluator_initialization()))
    test_results.append(("단일 모델 평가", test_single_model_evaluation()))
    test_results.append(("모델 비교", test_model_comparison()))
    test_results.append(("혼동 행렬 시각화", test_confusion_matrix_visualization()))
    test_results.append(("모델 비교 시각화", test_model_comparison_visualization()))
    test_results.append(("종합 평가 보고서 생성", test_evaluation_report_generation()))
    test_results.append(("보고서 저장/로딩", test_report_save_and_load()))
    test_results.append(("평가 요약 정보", test_evaluation_summary()))
    
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
        logger.info("✅ ModelEvaluator 클래스가 올바르게 구현되었습니다.")
        logger.info("✅ 모델 평가 및 비교 기능이 제대로 작동합니다.")
        logger.info("✅ 시각화 및 보고서 생성 기능이 정상 작동합니다.")
    else:
        logger.info("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("📊 모델 평가 모듈 테스트 완료 📊")