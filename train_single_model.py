#!/usr/bin/env python3
"""
단일 모델 학습 스크립트 - SVM 또는 Random Forest 중 하나만 학습
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG


def train_single_model(model_type='svm', skip_augmentation=True, use_existing_augmented=False):
    """
    단일 모델만 학습하고 평가합니다.
    
    Parameters:
    -----------
    model_type : str
        'svm' 또는 'random_forest'
    skip_augmentation : bool
        데이터 증강을 건너뛸지 여부 (빠른 테스트를 위해 기본값 True)
    use_existing_augmented : bool
        기존에 생성된 증강 데이터를 사용할지 여부
    """
    logger = setup_logger("SingleModelTraining", "INFO")
    
    logger.info(f"🎯 {model_type.upper()} 모델만 학습을 시작합니다.")
    logger.info("=" * 60)
    
    try:
        # 1. 데이터 로딩
        logger.info("📊 1단계: 데이터 로딩 중...")
        data_pipeline = DataPipeline(DEFAULT_CONFIG)
        X_train, y_train, X_val, y_val, X_test, y_test = data_pipeline.run_complete_pipeline(
            skip_augmentation=skip_augmentation,
            use_existing_augmented=use_existing_augmented
        )
        
        logger.info(f"✅ 데이터 로딩 완료:")
        logger.info(f"  훈련 데이터: {len(X_train)}개")
        logger.info(f"  검증 데이터: {len(X_val)}개")
        logger.info(f"  테스트 데이터: {len(X_test)}개")
        
        # 2. 모델 훈련
        logger.info(f"\n🤖 2단계: {model_type.upper()} 모델 훈련 중...")
        model_trainer = ModelTrainer(DEFAULT_CONFIG)
        
        # 단일 모델만 훈련
        training_result = model_trainer.train_single_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            cv_folds=5
        )
        
        logger.info(f"✅ 모델 훈련 완료:")
        logger.info(f"  최적 점수: {training_result.best_score:.4f}")
        logger.info(f"  최적 파라미터: {training_result.best_params}")
        logger.info(f"  훈련 시간: {training_result.training_time:.2f}초")
        
        # 3. 모델 평가
        logger.info(f"\n📈 3단계: 모델 평가 중...")
        model_evaluator = ModelEvaluator(DEFAULT_CONFIG)
        
        # 훈련된 모델 가져오기
        trained_model = model_trainer.trained_models[model_type]
        
        # 평가 수행
        eval_result = model_evaluator.evaluate_model(
            trained_model, X_test, y_test, model_type
        )
        
        logger.info(f"✅ 모델 평가 완료:")
        logger.info(f"  정확도: {eval_result.accuracy:.4f}")
        logger.info(f"  F1-score (macro): {eval_result.f1_macro:.4f}")
        logger.info(f"  정밀도 (macro): {eval_result.precision_macro:.4f}")
        logger.info(f"  재현율 (macro): {eval_result.recall_macro:.4f}")
        
        # 클래스별 성능 출력
        logger.info("\n📊 클래스별 성능:")
        for class_name in DEFAULT_CONFIG.class_names:
            precision = eval_result.class_precision.get(class_name, 0)
            recall = eval_result.class_recall.get(class_name, 0)
            f1 = eval_result.class_f1.get(class_name, 0)
            logger.info(f"  {class_name}:")
            logger.info(f"    정밀도: {precision:.4f}")
            logger.info(f"    재현율: {recall:.4f}")
            logger.info(f"    F1-score: {f1:.4f}")
        
        # 4. 모델 저장 (선택사항)
        save_model = input("\n💾 모델을 저장하시겠습니까? (y/N): ").strip().lower() == 'y'
        if save_model:
            logger.info("모델 저장 중...")
            saved_paths = model_trainer.save_models()
            logger.info(f"✅ 모델 저장 완료: {saved_paths}")
        
        # Random Forest의 경우 특징 중요도 출력
        if model_type == 'random_forest' and training_result.feature_importance:
            logger.info("\n🌟 상위 10개 중요 특징:")
            importance_pairs = model_trainer.get_feature_importance(model_type)
            if importance_pairs:
                for i, (feature, importance) in enumerate(importance_pairs[:10]):
                    logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        logger.info("\n🎉 단일 모델 학습 및 평가 완료!")
        
        return {
            'training_result': training_result,
            'evaluation_result': eval_result
        }
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="단일 모델 학습 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['svm', 'random_forest'],
        default='svm',
        help='학습할 모델 타입 (기본값: svm)'
    )
    
    parser.add_argument(
        '--with-augmentation',
        action='store_true',
        help='데이터 증강 포함 (기본값: 증강 없이 빠른 실행)'
    )
    
    parser.add_argument(
        '--use-existing-augmented',
        action='store_true',
        help='기존에 생성된 증강 데이터를 사용'
    )
    
    args = parser.parse_args()
    
    # 모델 학습 실행
    start_time = time.time()
    
    try:
        results = train_single_model(
            model_type=args.model,
            skip_augmentation=not args.with_augmentation,
            use_existing_augmented=args.use_existing_augmented
        )
        
        total_time = time.time() - start_time
        print(f"\n⏱️  전체 실행 시간: {total_time:.2f}초")
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()