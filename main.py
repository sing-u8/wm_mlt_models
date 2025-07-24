#!/usr/bin/env python3
"""
수박 소리 분류 시스템 메인 실행 파이프라인

이 스크립트는 전체 머신러닝 파이프라인을 조정하고 실행합니다:
1. 데이터 로딩 및 전처리
2. 모델 훈련
3. 모델 평가
4. 모델 저장 및 Core ML 변환

design.md의 명세에 따라 구현되었습니다.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 프로젝트 모듈 임포트
from src.data.pipeline import DataPipeline
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.ml.model_converter import ModelConverter
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG


class PipelineCheckpoint:
    """
    파이프라인 실행 상태를 저장하고 복원하는 체크포인트 관리 클래스.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        체크포인트 관리자를 초기화합니다.
        
        Parameters:
        -----------
        checkpoint_dir : str
            체크포인트 파일을 저장할 디렉토리
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pipeline_state.json"
        
    def save_checkpoint(self, step: str, data: Dict[str, Any], 
                       execution_time: float = None):
        """
        현재 파이프라인 상태를 체크포인트에 저장합니다.
        
        Parameters:
        -----------
        step : str
            현재 실행 단계
        data : Dict[str, Any]
            저장할 데이터
        execution_time : float, optional
            실행 시간
        """
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'data': data
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        저장된 체크포인트를 로드합니다.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            체크포인트 데이터 또는 None
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def clear_checkpoint(self):
        """체크포인트 파일을 삭제합니다."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class WatermelonClassificationPipeline:
    """
    수박 소리 분류 시스템의 메인 실행 파이프라인.
    
    이 클래스는 다음 단계들을 순차적으로 실행합니다:
    1. 데이터 로딩 및 증강
    2. 모델 훈련
    3. 모델 평가
    4. 모델 저장 및 형식 변환
    """
    
    def __init__(self, config=None, checkpoint_dir: str = "checkpoints"):
        """
        파이프라인을 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        checkpoint_dir : str
            체크포인트 디렉토리
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = setup_logger("WatermelonPipeline", "INFO")
        self.checkpoint_manager = PipelineCheckpoint(checkpoint_dir)
        
        # 파이프라인 구성요소 초기화
        self.data_pipeline = DataPipeline(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        self.model_converter = ModelConverter(self.config)
        
        # 실행 통계
        self.pipeline_start_time = None
        self.step_times = {}
        
        self.logger.info("🚀 수박 소리 분류 파이프라인 초기화 완료")
        self.logger.info(f"구성: {len(self.config.class_names)}개 클래스, "
                        f"샘플레이트 {self.config.sample_rate}Hz")
    
    def step_1_load_data(self, skip_augmentation: bool = False) -> Tuple:
        """
        1단계: 데이터 로딩 및 전처리를 수행합니다.
        
        Parameters:
        -----------
        skip_augmentation : bool
            데이터 증강을 건너뛸지 여부
            
        Returns:
        --------
        Tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        step_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("📊 1단계: 데이터 로딩 및 전처리 시작")
        self.logger.info("=" * 60)
        
        try:
            # 데이터 파이프라인 실행
            self.logger.info("데이터 파이프라인 실행 중...")
            
            pipeline_result = self.data_pipeline.run_complete_pipeline(
                skip_augmentation=skip_augmentation
            )
            
            # 결과 추출
            datasets = pipeline_result['datasets']
            X_train = datasets['train']['features']
            y_train = datasets['train']['labels']
            X_val = datasets['validation']['features']
            y_val = datasets['validation']['labels']
            X_test = datasets['test']['features']
            y_test = datasets['test']['labels']
            
            # 데이터 통계 로깅
            self.logger.info(f"✅ 데이터 로딩 완료:")
            self.logger.info(f"  훈련 데이터: {len(X_train)}개 샘플")
            self.logger.info(f"  검증 데이터: {len(X_val)}개 샘플")
            self.logger.info(f"  테스트 데이터: {len(X_test)}개 샘플")
            self.logger.info(f"  특징 차원: {X_train.shape[1] if len(X_train) > 0 else 'N/A'}")
            
            # 클래스 분포 확인
            import numpy as np
            if len(y_train) > 0:
                unique, counts = np.unique(y_train, return_counts=True)
                self.logger.info("  훈련 데이터 클래스 분포:")
                for i, (cls, count) in enumerate(zip(unique, counts)):
                    class_name = self.config.class_names[int(cls)] if cls < len(self.config.class_names) else f"Class_{cls}"
                    self.logger.info(f"    {class_name}: {count}개 ({count/len(y_train)*100:.1f}%)")
            
            step_time = time.time() - step_start_time
            self.step_times['data_loading'] = step_time
            
            # 체크포인트 저장
            checkpoint_data = {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_dim': X_train.shape[1] if len(X_train) > 0 else 0,
                'augmentation_skipped': skip_augmentation
            }
            self.checkpoint_manager.save_checkpoint('data_loading', checkpoint_data, step_time)
            
            self.logger.info(f"⏱️  1단계 완료 시간: {step_time:.2f}초")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"❌ 1단계 실패: {e}")
            raise
    
    def step_2_train_models(self, X_train, y_train, cv_folds: int = 5) -> Dict[str, Any]:
        """
        2단계: 모델 훈련을 수행합니다.
        
        Parameters:
        -----------
        X_train : array-like
            훈련 특징 데이터
        y_train : array-like
            훈련 레이블 데이터
        cv_folds : int
            교차 검증 폴드 수
            
        Returns:
        --------
        Dict[str, Any]
            훈련 결과
        """
        step_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("🤖 2단계: 모델 훈련 시작")
        self.logger.info("=" * 60)
        
        try:
            # 모델 훈련 실행
            self.logger.info(f"교차 검증 훈련 시작 ({cv_folds}-fold CV)")
            
            training_results = self.model_trainer.train_with_cv(
                X_train, y_train, cv_folds=cv_folds
            )
            
            # 훈련 결과 로깅
            self.logger.info("✅ 모델 훈련 완료:")
            
            for model_name, result in training_results.items():
                self.logger.info(f"  {model_name.upper()}:")
                self.logger.info(f"    최적 점수: {result.best_score:.4f}")
                self.logger.info(f"    최적 파라미터: {result.best_params}")
                self.logger.info(f"    CV 점수 평균: {result.cv_scores.mean():.4f} ± {result.cv_scores.std():.4f}")
                self.logger.info(f"    훈련 시간: {result.training_time:.2f}초")
            
            step_time = time.time() - step_start_time
            self.step_times['model_training'] = step_time
            
            # 체크포인트 저장
            checkpoint_data = {
                'models_trained': list(training_results.keys()),
                'best_scores': {name: result.best_score for name, result in training_results.items()},
                'cv_folds': cv_folds
            }
            self.checkpoint_manager.save_checkpoint('model_training', checkpoint_data, step_time)
            
            self.logger.info(f"⏱️  2단계 완료 시간: {step_time:.2f}초")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ 2단계 실패: {e}")
            raise
    
    def step_3_evaluate_models(self, X_test, y_test) -> Dict[str, Any]:
        """
        3단계: 모델 평가를 수행합니다.
        
        Parameters:
        -----------
        X_test : array-like
            테스트 특징 데이터
        y_test : array-like
            테스트 레이블 데이터
            
        Returns:
        --------
        Dict[str, Any]
            평가 결과
        """
        step_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("📈 3단계: 모델 평가 시작")
        self.logger.info("=" * 60)
        
        try:
            # 개별 모델 평가
            evaluation_results = {}
            
            for model_name, model in self.model_trainer.trained_models.items():
                self.logger.info(f"{model_name.upper()} 모델 평가 중...")
                
                eval_result = self.model_evaluator.evaluate_model(
                    model, X_test, y_test, model_name
                )
                
                evaluation_results[model_name] = eval_result
                
                # 평가 결과 로깅
                metrics = eval_result.classification_metrics
                self.logger.info(f"  정확도: {metrics.accuracy:.4f}")
                self.logger.info(f"  F1-score (macro): {metrics.f1_macro:.4f}")
                self.logger.info(f"  정밀도 (macro): {metrics.precision_macro:.4f}")
                self.logger.info(f"  재현율 (macro): {metrics.recall_macro:.4f}")
            
            # 모델 비교
            if len(self.model_trainer.trained_models) >= 2:
                self.logger.info("모델 간 성능 비교 수행 중...")
                
                model_names = list(self.model_trainer.trained_models.keys())
                model1_name, model2_name = model_names[0], model_names[1]
                
                comparison_result = self.model_evaluator.compare_models(
                    self.model_trainer.trained_models[model1_name],
                    self.model_trainer.trained_models[model2_name],
                    X_test, y_test, model1_name, model2_name
                )
                
                # 비교 결과 로깅
                self.logger.info("📊 모델 비교 결과:")
                self.logger.info(f"  {model1_name} vs {model2_name}")
                self.logger.info(f"  정확도 차이: {comparison_result.accuracy_difference:.4f}")
                self.logger.info(f"  F1-score 차이: {comparison_result.f1_difference:.4f}")
                self.logger.info(f"  통계적 유의성 (정확도): p={comparison_result.accuracy_p_value:.4f}")
                
                evaluation_results['comparison'] = comparison_result
            
            # 종합 평가 보고서 생성
            evaluation_report = self.model_evaluator.create_evaluation_report(
                evaluation_results, save_report=True
            )
            
            step_time = time.time() - step_start_time
            self.step_times['model_evaluation'] = step_time
            
            # 체크포인트 저장
            checkpoint_data = {
                'models_evaluated': list(evaluation_results.keys()),
                'best_model': max(evaluation_results.keys(), 
                                key=lambda k: evaluation_results[k].classification_metrics.accuracy 
                                if k != 'comparison' else 0),
                'evaluation_completed': True
            }
            self.checkpoint_manager.save_checkpoint('model_evaluation', checkpoint_data, step_time)
            
            self.logger.info(f"✅ 평가 보고서 저장: {evaluation_report.report_path}")
            self.logger.info(f"⏱️  3단계 완료 시간: {step_time:.2f}초")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ 3단계 실패: {e}")
            raise
    
    def step_4_save_and_convert_models(self, convert_to_coreml: bool = True) -> Dict[str, Any]:
        """
        4단계: 모델 저장 및 형식 변환을 수행합니다.
        
        Parameters:
        -----------
        convert_to_coreml : bool
            Core ML 형식으로 변환할지 여부
            
        Returns:
        --------
        Dict[str, Any]
            변환 결과
        """
        step_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("💾 4단계: 모델 저장 및 변환 시작")
        self.logger.info("=" * 60)
        
        try:
            conversion_results = {}
            
            for model_name, model in self.model_trainer.trained_models.items():
                self.logger.info(f"{model_name.upper()} 모델 저장 중...")
                
                # Pickle 형식으로 저장
                model_metadata = {
                    'model_type': model_name,
                    'feature_count': 30,
                    'class_names': self.config.class_names,
                    'training_completed': True
                }
                
                pickle_path = self.model_converter.save_pickle_model(
                    model, model_name, model_metadata
                )
                
                self.logger.info(f"✅ Pickle 모델 저장: {pickle_path}")
                
                # Core ML 변환 (요청된 경우)
                if convert_to_coreml:
                    try:
                        self.logger.info(f"{model_name.upper()} Core ML 변환 중...")
                        
                        conversion_result = self.model_converter.convert_model_with_validation(
                            model, model_name, validate=True
                        )
                        
                        conversion_results[model_name] = conversion_result
                        
                        self.logger.info(f"✅ Core ML 변환 완료:")
                        self.logger.info(f"  파일: {conversion_result.converted_path}")
                        self.logger.info(f"  크기: {conversion_result.file_size_bytes:,} bytes")
                        self.logger.info(f"  변환 시간: {conversion_result.conversion_time:.3f}초")
                        self.logger.info(f"  검증 통과: {conversion_result.validation_passed}")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ {model_name} Core ML 변환 실패: {e}")
                        self.logger.warning("Pickle 형식 모델은 정상적으로 저장되었습니다.")
            
            # 변환 요약 생성
            conversion_summary = self.model_converter.get_conversion_summary()
            
            step_time = time.time() - step_start_time
            self.step_times['model_conversion'] = step_time
            
            # 체크포인트 저장
            checkpoint_data = {
                'pickle_models_saved': len(self.model_trainer.trained_models),
                'coreml_conversions': len(conversion_results),
                'conversion_success_rate': conversion_summary.get('success_rate', 0),
                'conversion_completed': True
            }
            self.checkpoint_manager.save_checkpoint('model_conversion', checkpoint_data, step_time)
            
            self.logger.info(f"📊 변환 요약:")
            self.logger.info(f"  총 변환: {conversion_summary.get('total_conversions', 0)}개")
            self.logger.info(f"  성공률: {conversion_summary.get('success_rate', 0):.1%}")
            self.logger.info(f"⏱️  4단계 완료 시간: {step_time:.2f}초")
            
            return {
                'conversion_results': conversion_results,
                'conversion_summary': conversion_summary
            }
            
        except Exception as e:
            self.logger.error(f"❌ 4단계 실패: {e}")
            raise
    
    def run_complete_pipeline(self, skip_augmentation: bool = False,
                            cv_folds: int = 5, convert_to_coreml: bool = True,
                            resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        전체 파이프라인을 실행합니다.
        
        Parameters:
        -----------
        skip_augmentation : bool
            데이터 증강을 건너뛸지 여부
        cv_folds : int
            교차 검증 폴드 수
        convert_to_coreml : bool
            Core ML 변환 수행 여부
        resume_from_checkpoint : bool
            체크포인트에서 재시작할지 여부
            
        Returns:
        --------
        Dict[str, Any]
            전체 파이프라인 실행 결과
        """
        self.pipeline_start_time = time.time()
        
        self.logger.info("🎯" * 20)
        self.logger.info("🚀 수박 소리 분류 파이프라인 시작 🚀")
        self.logger.info("🎯" * 20)
        self.logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"구성: 증강건너뛰기={skip_augmentation}, CV폴드={cv_folds}, CoreML변환={convert_to_coreml}")
        
        # 체크포인트 확인
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                self.logger.info(f"📋 체크포인트에서 재시작: {checkpoint['step']} 단계부터")
        
        try:
            pipeline_results = {}
            
            # 1단계: 데이터 로딩
            if not checkpoint or checkpoint['step'] in ['data_loading']:
                X_train, y_train, X_val, y_val, X_test, y_test = self.step_1_load_data(skip_augmentation)
                pipeline_results['data_loading'] = {
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                }
            else:
                self.logger.info("📋 데이터 로딩 단계 건너뛰기 (체크포인트에서 복원)")
                # 실제 구현에서는 체크포인트에서 데이터를 복원해야 함
                # 여기서는 간단화를 위해 다시 로딩
                X_train, y_train, X_val, y_val, X_test, y_test = self.step_1_load_data(skip_augmentation)
            
            # 2단계: 모델 훈련
            if not checkpoint or checkpoint['step'] in ['data_loading', 'model_training']:
                training_results = self.step_2_train_models(X_train, y_train, cv_folds)
                pipeline_results['model_training'] = training_results
            else:
                self.logger.info("📋 모델 훈련 단계 건너뛰기 (체크포인트에서 복원)")
            
            # 3단계: 모델 평가
            if not checkpoint or checkpoint['step'] in ['data_loading', 'model_training', 'model_evaluation']:
                evaluation_results = self.step_3_evaluate_models(X_test, y_test)
                pipeline_results['model_evaluation'] = evaluation_results
            else:
                self.logger.info("📋 모델 평가 단계 건너뛰기 (체크포인트에서 복원)")
            
            # 4단계: 모델 변환
            conversion_results = self.step_4_save_and_convert_models(convert_to_coreml)
            pipeline_results['model_conversion'] = conversion_results
            
            # 전체 파이프라인 완료
            total_time = time.time() - self.pipeline_start_time
            
            self.logger.info("🎉" * 20)
            self.logger.info("✅ 파이프라인 완료! ✅")
            self.logger.info("🎉" * 20)
            self.logger.info(f"전체 실행 시간: {total_time:.2f}초")
            
            # 단계별 시간 요약
            if self.step_times:
                self.logger.info("📊 단계별 실행 시간:")
                for step, duration in self.step_times.items():
                    self.logger.info(f"  {step}: {duration:.2f}초 ({duration/total_time*100:.1f}%)")
            
            # 최종 결과 요약
            pipeline_results['execution_summary'] = {
                'total_time': total_time,
                'step_times': self.step_times,
                'completed_at': datetime.now().isoformat(),
                'success': True
            }
            
            # 체크포인트 정리
            self.checkpoint_manager.clear_checkpoint()
            
            return pipeline_results
            
        except Exception as e:
            total_time = time.time() - self.pipeline_start_time
            self.logger.error(f"💥 파이프라인 실패: {e}")
            self.logger.error(f"실행 시간: {total_time:.2f}초")
            
            # 실패 시에도 체크포인트는 유지 (재시작 가능)
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        현재 파이프라인 상태를 반환합니다.
        
        Returns:
        --------
        Dict[str, Any]
            파이프라인 상태 정보
        """
        checkpoint = self.checkpoint_manager.load_checkpoint()
        
        status = {
            'pipeline_initialized': True,
            'checkpoint_available': checkpoint is not None,
            'components_ready': {
                'data_pipeline': self.data_pipeline is not None,
                'model_trainer': self.model_trainer is not None,
                'model_evaluator': self.model_evaluator is not None,
                'model_converter': self.model_converter is not None
            }
        }
        
        if checkpoint:
            status.update({
                'last_completed_step': checkpoint['step'],
                'last_execution_time': checkpoint.get('execution_time'),
                'last_timestamp': checkpoint['timestamp']
            })
        
        return status


def create_argument_parser():
    """명령줄 인수 파서를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="수박 소리 분류 시스템 메인 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                              # 기본 설정으로 전체 파이프라인 실행
  python main.py --skip-augmentation          # 데이터 증강 없이 실행
  python main.py --cv-folds 10                # 10-fold 교차 검증 사용
  python main.py --no-coreml                  # Core ML 변환 건너뛰기
  python main.py --resume                     # 체크포인트에서 재시작
  python main.py --status                     # 파이프라인 상태만 확인
        """
    )
    
    # 실행 옵션
    parser.add_argument(
        '--skip-augmentation', 
        action='store_true',
        help='데이터 증강을 건너뛰고 원본 데이터만 사용'
    )
    
    parser.add_argument(
        '--cv-folds', 
        type=int, 
        default=5,
        help='교차 검증 폴드 수 (기본값: 5)'
    )
    
    parser.add_argument(
        '--no-coreml', 
        action='store_true',
        help='Core ML 변환을 건너뛰기'
    )
    
    # 체크포인트 관련
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='이전 체크포인트에서 파이프라인 재시작'
    )
    
    parser.add_argument(
        '--checkpoint-dir', 
        type=str, 
        default='checkpoints',
        help='체크포인트 저장 디렉토리 (기본값: checkpoints)'
    )
    
    parser.add_argument(
        '--clear-checkpoint', 
        action='store_true',
        help='기존 체크포인트 삭제'
    )
    
    # 정보 확인
    parser.add_argument(
        '--status', 
        action='store_true',
        help='파이프라인 상태 확인만 수행'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='실제 실행 없이 설정만 확인'
    )
    
    # 로깅 옵션
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='로그 레벨 설정 (기본값: INFO)'
    )
    
    return parser


def main():
    """메인 실행 함수"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 로거 설정
    logger = setup_logger("MainPipeline", args.log_level)
    
    try:
        # 파이프라인 초기화
        pipeline = WatermelonClassificationPipeline(
            checkpoint_dir=args.checkpoint_dir
        )
        
        # 체크포인트 정리 (요청된 경우)
        if args.clear_checkpoint:
            pipeline.checkpoint_manager.clear_checkpoint()
            logger.info("✅ 체크포인트가 삭제되었습니다.")
            return
        
        # 파이프라인 상태 확인 (요청된 경우)
        if args.status:
            status = pipeline.get_pipeline_status()
            logger.info("📊 파이프라인 상태:")
            logger.info(f"  초기화됨: {status['pipeline_initialized']}")
            logger.info(f"  체크포인트 사용 가능: {status['checkpoint_available']}")
            
            if status['checkpoint_available']:
                logger.info(f"  마지막 완료 단계: {status['last_completed_step']}")
                logger.info(f"  마지막 실행 시간: {status.get('last_execution_time', 'N/A')}초")
                logger.info(f"  타임스탬프: {status['last_timestamp']}")
            
            components = status['components_ready']
            logger.info("  구성요소 준비 상태:")
            for component, ready in components.items():
                status_icon = "✅" if ready else "❌"
                logger.info(f"    {component}: {status_icon}")
            
            return
        
        # Dry run (설정 확인만)
        if args.dry_run:
            logger.info("🔍 Dry run 모드: 설정 확인만 수행")
            logger.info(f"  데이터 증강 건너뛰기: {args.skip_augmentation}")
            logger.info(f"  교차 검증 폴드: {args.cv_folds}")
            logger.info(f"  Core ML 변환: {not args.no_coreml}")
            logger.info(f"  체크포인트에서 재시작: {args.resume}")
            logger.info(f"  체크포인트 디렉토리: {args.checkpoint_dir}")
            logger.info("✅ 설정 확인 완료")
            return
        
        # 전체 파이프라인 실행
        logger.info("🚀 수박 소리 분류 파이프라인 시작")
        
        results = pipeline.run_complete_pipeline(
            skip_augmentation=args.skip_augmentation,
            cv_folds=args.cv_folds,
            convert_to_coreml=not args.no_coreml,
            resume_from_checkpoint=args.resume
        )
        
        # 최종 결과 로깅
        execution_summary = results.get('execution_summary', {})
        logger.info("🎉 파이프라인 실행 완료!")
        logger.info(f"총 실행 시간: {execution_summary.get('total_time', 0):.2f}초")
        logger.info(f"완료 시간: {execution_summary.get('completed_at', 'N/A')}")
        
        logger.info("🎯 다음 단계:")
        logger.info("  1. results/ 디렉토리에서 평가 보고서 확인")
        logger.info("  2. models/ 디렉토리에서 저장된 모델 확인")
        logger.info("  3. docs/COREML_USAGE.md에서 Core ML 모델 사용법 확인")
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 파이프라인이 중단되었습니다.")
        logger.info("💡 --resume 옵션으로 체크포인트에서 재시작할 수 있습니다.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 파이프라인 실행 중 오류 발생: {e}")
        logger.error("💡 --resume 옵션으로 체크포인트에서 재시작해보세요.")
        logger.error("💡 --status 옵션으로 현재 상태를 확인할 수 있습니다.")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()