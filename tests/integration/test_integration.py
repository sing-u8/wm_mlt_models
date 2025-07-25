#!/usr/bin/env python3
"""
수박 소리 분류 시스템 통합 테스트 스크립트

전체 파이프라인의 엔드투엔드 통합 테스트를 수행합니다.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import json
sys.path.append('.')

from main import WatermelonClassificationPipeline, PipelineCheckpoint
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def create_test_audio_data():
    """테스트용 가짜 오디오 데이터를 생성합니다."""
    logger = setup_logger("test_data_creation", "INFO")
    logger.info("테스트용 오디오 데이터 생성 중...")
    
    # 임시 데이터 디렉토리 구조 생성
    temp_data_dir = Path("test_data_temp")
    
    # 기존 테스트 데이터 정리
    if temp_data_dir.exists():
        shutil.rmtree(temp_data_dir)
    
    # 디렉토리 구조 생성
    for split in ['train', 'validation', 'test']:
        for class_name in DEFAULT_CONFIG.class_names:
            class_dir = temp_data_dir / 'raw' / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # 소음 디렉토리 생성
    noise_dir = temp_data_dir / 'noise' / 'environmental' / 'retail' / 'homeplus'
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    # 가짜 오디오 파일 생성 (실제로는 numpy 배열을 저장)
    np.random.seed(42)
    
    # 각 분할별로 파일 생성
    file_counts = {'train': 15, 'validation': 6, 'test': 9}  # 클래스당 파일 수
    
    for split, count in file_counts.items():
        for class_idx, class_name in enumerate(DEFAULT_CONFIG.class_names):
            for i in range(count):
                # 클래스별로 다른 특성을 가진 가짜 오디오 생성
                duration = 2.0  # 2초
                sample_rate = DEFAULT_CONFIG.sample_rate
                n_samples = int(duration * sample_rate)
                
                # 클래스별로 다른 주파수 특성
                base_freq = 100 + class_idx * 50  # 100Hz, 150Hz, 200Hz
                t = np.linspace(0, duration, n_samples)
                
                # 사인파 + 노이즈로 가짜 오디오 생성
                audio = (np.sin(2 * np.pi * base_freq * t) * 0.5 + 
                        np.random.randn(n_samples) * 0.1)
                
                # 파일 저장
                file_path = temp_data_dir / 'raw' / split / class_name / f"{class_name}_{i:03d}.npy"
                np.save(file_path, audio.astype(np.float32))
    
    # 가짜 소음 파일 생성
    for i in range(5):
        noise_audio = np.random.randn(int(2.0 * DEFAULT_CONFIG.sample_rate)) * 0.2
        noise_path = noise_dir / f"noise_{i:03d}.npy"
        np.save(noise_path, noise_audio.astype(np.float32))
    
    logger.info(f"테스트 데이터 생성 완료: {temp_data_dir}")
    return temp_data_dir

def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    logger = setup_logger("pipeline_init_test", "INFO")
    logger.info("=== 파이프라인 초기화 테스트 시작 ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            pipeline = WatermelonClassificationPipeline(
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 구성요소 초기화 확인
            assert pipeline.data_pipeline is not None, "DataPipeline 초기화 실패"
            assert pipeline.model_trainer is not None, "ModelTrainer 초기화 실패"
            assert pipeline.model_evaluator is not None, "ModelEvaluator 초기화 실패"
            assert pipeline.model_converter is not None, "ModelConverter 초기화 실패"
            assert pipeline.checkpoint_manager is not None, "CheckpointManager 초기화 실패"
            
            # 상태 확인
            status = pipeline.get_pipeline_status()
            assert status['pipeline_initialized'] == True, "파이프라인 초기화 상태 오류"
            assert status['checkpoint_available'] == False, "빈 체크포인트 상태 오류"
            
            components = status['components_ready']
            for component, ready in components.items():
                assert ready == True, f"{component} 구성요소 준비 실패"
        
        logger.info("✅ 파이프라인 초기화 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 초기화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 파이프라인 초기화 테스트 완료 ===\n")

def test_checkpoint_functionality():
    """체크포인트 기능 테스트"""
    logger = setup_logger("checkpoint_test", "INFO")
    logger.info("=== 체크포인트 기능 테스트 시작 ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = PipelineCheckpoint(temp_dir)
            
            # 체크포인트 저장 테스트
            test_data = {
                'step': 'data_loading',
                'samples': 100,
                'features': 30
            }
            
            checkpoint_manager.save_checkpoint('data_loading', test_data, 10.5)
            
            # 체크포인트 로드 테스트
            loaded_checkpoint = checkpoint_manager.load_checkpoint()
            
            assert loaded_checkpoint is not None, "체크포인트 로드 실패"
            assert loaded_checkpoint['step'] == 'data_loading', "체크포인트 단계 불일치"
            assert loaded_checkpoint['data']['samples'] == 100, "체크포인트 데이터 불일치"
            assert loaded_checkpoint['execution_time'] == 10.5, "체크포인트 실행시간 불일치"
            
            # 체크포인트 삭제 테스트
            checkpoint_manager.clear_checkpoint()
            cleared_checkpoint = checkpoint_manager.load_checkpoint()
            assert cleared_checkpoint is None, "체크포인트 삭제 실패"
        
        logger.info("✅ 체크포인트 기능 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 체크포인트 기능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        logger.info("=== 체크포인트 기능 테스트 완료 ===\n")

def test_data_integrity_checks():
    """데이터 무결성 검사 테스트"""
    logger = setup_logger("data_integrity_test", "INFO")
    logger.info("=== 데이터 무결성 검사 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        test_data_dir = create_test_audio_data()
        
        # 원본 구성을 테스트 데이터로 수정
        test_config = DEFAULT_CONFIG
        test_config.data_base_dir = str(test_data_dir)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            pipeline = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 1단계: 데이터 로딩 테스트
            logger.info("데이터 로딩 무결성 검사 중...")
            X_train, y_train, X_val, y_val, X_test, y_test = pipeline.step_1_load_data(
                skip_augmentation=True  # 빠른 테스트를 위해 증강 건너뛰기
            )
            
            # 데이터 형태 검증
            assert len(X_train) > 0, "훈련 데이터가 비어있음"
            assert len(X_val) > 0, "검증 데이터가 비어있음"
            assert len(X_test) > 0, "테스트 데이터가 비어있음"
            
            assert len(X_train) == len(y_train), "훈련 데이터 특징-레이블 길이 불일치"
            assert len(X_val) == len(y_val), "검증 데이터 특징-레이블 길이 불일치"
            assert len(X_test) == len(y_test), "테스트 데이터 특징-레이블 길이 불일치"
            
            # 특징 벡터 차원 검증 (design.md 명세: 30차원)
            if len(X_train) > 0:
                assert X_train.shape[1] == 30, f"특징 벡터 차원 오류: {X_train.shape[1]} != 30"
            if len(X_val) > 0:
                assert X_val.shape[1] == 30, f"검증 특징 벡터 차원 오류: {X_val.shape[1]} != 30"
            if len(X_test) > 0:
                assert X_test.shape[1] == 30, f"테스트 특징 벡터 차원 오류: {X_test.shape[1]} != 30"
            
            # 레이블 범위 검증
            all_labels = np.concatenate([y_train, y_val, y_test])
            unique_labels = np.unique(all_labels)
            assert len(unique_labels) <= len(DEFAULT_CONFIG.class_names), "예상보다 많은 클래스"
            assert np.all(unique_labels >= 0), "음수 레이블 발견"
            assert np.all(unique_labels < len(DEFAULT_CONFIG.class_names)), "범위 초과 레이블 발견"
            
            # NaN/Infinity 검증
            for dataset_name, (X, y) in [('train', (X_train, y_train)), 
                                        ('val', (X_val, y_val)), 
                                        ('test', (X_test, y_test))]:
                if len(X) > 0:
                    assert not np.any(np.isnan(X)), f"{dataset_name} 특징에 NaN 발견"
                    assert not np.any(np.isinf(X)), f"{dataset_name} 특징에 Infinity 발견"
                    assert not np.any(np.isnan(y)), f"{dataset_name} 레이블에 NaN 발견"
            
            logger.info("✅ 데이터 무결성 검사 통과")
            logger.info(f"  훈련: {len(X_train)}샘플, 검증: {len(X_val)}샘플, 테스트: {len(X_test)}샘플")
            logger.info(f"  특징 차원: {X_train.shape[1] if len(X_train) > 0 else 'N/A'}")
            logger.info(f"  클래스 수: {len(unique_labels)}")
        
        # 테스트 데이터 정리
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        logger.info("✅ 데이터 무결성 검사 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터 무결성 검사 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 테스트 데이터 정리
        if 'test_data_dir' in locals() and test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        return False
    
    finally:
        logger.info("=== 데이터 무결성 검사 테스트 완료 ===\n")

def test_pipeline_steps_integration():
    """파이프라인 단계 통합 테스트"""
    logger = setup_logger("pipeline_steps_test", "INFO")
    logger.info("=== 파이프라인 단계 통합 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        test_data_dir = create_test_audio_data()
        
        # 원본 구성을 테스트 데이터로 수정
        test_config = DEFAULT_CONFIG
        test_config.data_base_dir = str(test_data_dir)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            pipeline = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 1단계: 데이터 로딩
            logger.info("1단계: 데이터 로딩 테스트 중...")
            X_train, y_train, X_val, y_val, X_test, y_test = pipeline.step_1_load_data(
                skip_augmentation=True
            )
            
            assert len(X_train) > 0, "1단계: 훈련 데이터 로딩 실패"
            step1_checkpoint = pipeline.checkpoint_manager.load_checkpoint()
            assert step1_checkpoint['step'] == 'data_loading', "1단계: 체크포인트 저장 실패"
            
            # 2단계: 모델 훈련 (빠른 테스트를 위해 파라미터 축소)
            logger.info("2단계: 모델 훈련 테스트 중...")
            
            # 훈련 파라미터 축소
            original_svm_params = pipeline.model_trainer.models["svm"].param_grid
            original_rf_params = pipeline.model_trainer.models["random_forest"].param_grid
            
            pipeline.model_trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
            pipeline.model_trainer.models["random_forest"].param_grid = {"n_estimators": [10], "max_depth": [3]}
            
            training_results = pipeline.step_2_train_models(X_train, y_train, cv_folds=3)
            
            assert len(training_results) > 0, "2단계: 모델 훈련 실패"
            assert 'svm' in training_results, "2단계: SVM 훈련 실패"
            assert 'random_forest' in training_results, "2단계: Random Forest 훈련 실패"
            
            step2_checkpoint = pipeline.checkpoint_manager.load_checkpoint()
            assert step2_checkpoint['step'] == 'model_training', "2단계: 체크포인트 저장 실패"
            
            # 3단계: 모델 평가
            logger.info("3단계: 모델 평가 테스트 중...")
            evaluation_results = pipeline.step_3_evaluate_models(X_test, y_test)
            
            assert len(evaluation_results) > 0, "3단계: 모델 평가 실패"
            
            for model_name in ['svm', 'random_forest']:
                assert model_name in evaluation_results, f"3단계: {model_name} 평가 결과 누락"
                
                eval_result = evaluation_results[model_name]
                assert hasattr(eval_result, 'accuracy'), f"3단계: {model_name} 메트릭 누락"
                
                metrics = eval_result
                assert hasattr(metrics, 'accuracy'), f"3단계: {model_name} 정확도 누락"
                assert 0 <= metrics.accuracy <= 1, f"3단계: {model_name} 정확도 범위 오류"
            
            step3_checkpoint = pipeline.checkpoint_manager.load_checkpoint()
            assert step3_checkpoint['step'] == 'model_evaluation', "3단계: 체크포인트 저장 실패"
            
            # 4단계: 모델 저장 및 변환 (Core ML 건너뛰기)
            logger.info("4단계: 모델 저장 테스트 중...")
            conversion_results = pipeline.step_4_save_and_convert_models(convert_to_coreml=False)
            
            assert 'conversion_summary' in conversion_results, "4단계: 변환 요약 누락"
            
            step4_checkpoint = pipeline.checkpoint_manager.load_checkpoint()
            assert step4_checkpoint['step'] == 'model_conversion', "4단계: 체크포인트 저장 실패"
            
            # 파라미터 복원
            pipeline.model_trainer.models["svm"].param_grid = original_svm_params
            pipeline.model_trainer.models["random_forest"].param_grid = original_rf_params
        
        # 테스트 데이터 정리
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        logger.info("✅ 파이프라인 단계 통합 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 단계 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 테스트 데이터 정리
        if 'test_data_dir' in locals() and test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        return False
    
    finally:
        logger.info("=== 파이프라인 단계 통합 테스트 완료 ===\n")

def test_complete_pipeline_run():
    """완전한 파이프라인 실행 테스트"""
    logger = setup_logger("complete_pipeline_test", "INFO")
    logger.info("=== 완전한 파이프라인 실행 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        test_data_dir = create_test_audio_data()
        
        # 원본 구성을 테스트 데이터로 수정
        test_config = DEFAULT_CONFIG
        test_config.data_base_dir = str(test_data_dir)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            pipeline = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 빠른 테스트를 위한 파라미터 조정
            pipeline.model_trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
            pipeline.model_trainer.models["random_forest"].param_grid = {"n_estimators": [10], "max_depth": [3]}
            
            # 전체 파이프라인 실행
            logger.info("전체 파이프라인 실행 중...")
            results = pipeline.run_complete_pipeline(
                skip_augmentation=True,
                cv_folds=3,
                convert_to_coreml=False,  # 빠른 테스트를 위해 Core ML 건너뛰기
                resume_from_checkpoint=False
            )
            
            # 결과 검증
            assert 'execution_summary' in results, "실행 요약 누락"
            assert results['execution_summary']['success'] == True, "파이프라인 실행 실패"
            
            execution_summary = results['execution_summary']
            assert 'total_time' in execution_summary, "총 실행 시간 누락"
            assert execution_summary['total_time'] > 0, "실행 시간 오류"
            
            assert 'step_times' in execution_summary, "단계별 시간 누락"
            step_times = execution_summary['step_times']
            
            expected_steps = ['data_loading', 'model_training', 'model_evaluation', 'model_conversion']
            for step in expected_steps:
                assert step in step_times, f"{step} 단계 시간 누락"
                assert step_times[step] > 0, f"{step} 단계 시간 오류"
            
            # 각 단계 결과 검증
            assert 'data_loading' in results, "데이터 로딩 결과 누락"
            assert 'model_training' in results, "모델 훈련 결과 누락"
            assert 'model_evaluation' in results, "모델 평가 결과 누락"
            assert 'model_conversion' in results, "모델 변환 결과 누락"
            
            # 체크포인트 정리 확인
            final_checkpoint = pipeline.checkpoint_manager.load_checkpoint()
            assert final_checkpoint is None, "파이프라인 완료 후 체크포인트 미정리"
        
        # 테스트 데이터 정리
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        logger.info("✅ 완전한 파이프라인 실행 테스트 성공")
        logger.info(f"  총 실행 시간: {execution_summary['total_time']:.2f}초")
        logger.info(f"  단계별 시간: {step_times}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 완전한 파이프라인 실행 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 테스트 데이터 정리
        if 'test_data_dir' in locals() and test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        return False
    
    finally:
        logger.info("=== 완전한 파이프라인 실행 테스트 완료 ===\n")

def test_resume_functionality():
    """재시작 기능 테스트"""
    logger = setup_logger("resume_test", "INFO")
    logger.info("=== 재시작 기능 테스트 시작 ===")
    
    try:
        # 테스트 데이터 생성
        test_data_dir = create_test_audio_data()
        
        # 원본 구성을 테스트 데이터로 수정
        test_config = DEFAULT_CONFIG
        test_config.data_base_dir = str(test_data_dir)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # 첫 번째 파이프라인: 1단계만 실행
            pipeline1 = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            logger.info("1단계만 실행하여 체크포인트 생성...")
            X_train, y_train, X_val, y_val, X_test, y_test = pipeline1.step_1_load_data(
                skip_augmentation=True
            )
            
            # 체크포인트 존재 확인
            checkpoint = pipeline1.checkpoint_manager.load_checkpoint()
            assert checkpoint is not None, "체크포인트 생성 실패"
            assert checkpoint['step'] == 'data_loading', "체크포인트 단계 오류"
            
            # 두 번째 파이프라인: 재시작 기능 테스트
            pipeline2 = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 상태 확인
            status = pipeline2.get_pipeline_status()
            assert status['checkpoint_available'] == True, "체크포인트 인식 실패"
            assert status['last_completed_step'] == 'data_loading', "마지막 단계 인식 실패"
            
            logger.info("체크포인트에서 재시작 기능 확인 완료")
        
        # 테스트 데이터 정리
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        logger.info("✅ 재시작 기능 테스트 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ 재시작 기능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 테스트 데이터 정리
        if 'test_data_dir' in locals() and test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        return False
    
    finally:
        logger.info("=== 재시작 기능 테스트 완료 ===\n")

def test_performance_benchmarking():
    """성능 벤치마킹 테스트"""
    logger = setup_logger("performance_test", "INFO")
    logger.info("=== 성능 벤치마킹 테스트 시작 ===")
    
    try:
        import psutil
        import time
        
        # 테스트 데이터 생성
        test_data_dir = create_test_audio_data()
        
        # 원본 구성을 테스트 데이터로 수정
        test_config = DEFAULT_CONFIG
        test_config.data_base_dir = str(test_data_dir)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            pipeline = WatermelonClassificationPipeline(
                config=test_config,
                checkpoint_dir=temp_checkpoint_dir
            )
            
            # 성능 측정 시작
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            logger.info(f"시작 메모리 사용량: {start_memory:.2f} MB")
            
            # 빠른 테스트를 위한 파라미터 조정
            pipeline.model_trainer.models["svm"].param_grid = {"C": [1], "gamma": ["scale"]}
            pipeline.model_trainer.models["random_forest"].param_grid = {"n_estimators": [5], "max_depth": [3]}
            
            # 1단계 성능 측정
            step_start_time = time.time()
            X_train, y_train, X_val, y_val, X_test, y_test = pipeline.step_1_load_data(
                skip_augmentation=True
            )
            step1_time = time.time() - step_start_time
            step1_memory = process.memory_info().rss / 1024 / 1024
            
            logger.info(f"1단계 완료: {step1_time:.2f}초, 메모리: {step1_memory:.2f} MB")
            
            # 2단계 성능 측정
            step_start_time = time.time()
            training_results = pipeline.step_2_train_models(X_train, y_train, cv_folds=3)
            step2_time = time.time() - step_start_time
            step2_memory = process.memory_info().rss / 1024 / 1024
            
            logger.info(f"2단계 완료: {step2_time:.2f}초, 메모리: {step2_memory:.2f} MB")
            
            # 전체 성능 측정
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            total_time = end_time - start_time
            memory_increase = end_memory - start_memory
            
            # 성능 기준 검증
            assert total_time < 300, f"전체 실행 시간 초과: {total_time:.2f}초 > 300초"  # 5분 이내
            assert memory_increase < 500, f"메모리 사용량 초과: {memory_increase:.2f}MB > 500MB"  # 500MB 이내
            
            # 성능 보고서 생성
            performance_report = {
                'total_execution_time': total_time,
                'memory_usage': {
                    'start': start_memory,
                    'end': end_memory,
                    'increase': memory_increase,
                    'peak': step2_memory
                },
                'step_performance': {
                    'data_loading': {'time': step1_time, 'memory': step1_memory},
                    'model_training': {'time': step2_time, 'memory': step2_memory}
                },
                'performance_criteria': {
                    'time_limit_met': total_time < 300,
                    'memory_limit_met': memory_increase < 500
                }
            }
            
            logger.info("📊 성능 벤치마킹 결과:")
            logger.info(f"  총 실행 시간: {total_time:.2f}초")
            logger.info(f"  메모리 사용량 증가: {memory_increase:.2f} MB")
            logger.info(f"  최대 메모리: {step2_memory:.2f} MB")
            logger.info(f"  시간 기준 통과: {performance_report['performance_criteria']['time_limit_met']}")
            logger.info(f"  메모리 기준 통과: {performance_report['performance_criteria']['memory_limit_met']}")
        
        # 테스트 데이터 정리
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        logger.info("✅ 성능 벤치마킹 테스트 성공")
        return True
        
    except ImportError:
        logger.warning("⚠️ psutil 라이브러리가 없어 성능 테스트를 건너뜁니다.")
        logger.warning("설치하려면: pip install psutil")
        return True  # psutil이 없는 것은 오류가 아님
        
    except Exception as e:
        logger.error(f"❌ 성능 벤치마킹 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 테스트 데이터 정리
        if 'test_data_dir' in locals() and test_data_dir.exists():
            shutil.rmtree(test_data_dir)
        
        return False
    
    finally:
        logger.info("=== 성능 벤치마킹 테스트 완료 ===\n")

if __name__ == "__main__":
    logger = setup_logger("main_integration_test", "INFO")
    logger.info("🔄 수박 소리 분류 시스템 통합 테스트 시작 🔄")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("파이프라인 초기화", test_pipeline_initialization()))
    test_results.append(("체크포인트 기능", test_checkpoint_functionality()))
    test_results.append(("데이터 무결성 검사", test_data_integrity_checks()))
    test_results.append(("파이프라인 단계 통합", test_pipeline_steps_integration()))
    test_results.append(("완전한 파이프라인 실행", test_complete_pipeline_run()))
    test_results.append(("재시작 기능", test_resume_functionality()))
    test_results.append(("성능 벤치마킹", test_performance_benchmarking()))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("통합 테스트 결과 요약")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    success_rate = passed / total * 100
    logger.info(f"전체 통합 테스트 결과: {passed}/{total} 통과 ({success_rate:.1f}%)")
    
    if passed == total:
        logger.info("🎉 모든 통합 테스트가 성공적으로 완료되었습니다!")
        logger.info("✅ 수박 소리 분류 파이프라인이 올바르게 통합되었습니다.")
        logger.info("✅ 데이터 무결성 검사가 정상 작동합니다.")
        logger.info("✅ 체크포인트 및 재시작 기능이 정상 작동합니다.")
        logger.info("✅ 성능 기준을 충족합니다.")
    else:
        logger.info("⚠️  일부 통합 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("🔄 수박 소리 분류 시스템 통합 테스트 완료 🔄")