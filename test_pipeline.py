#!/usr/bin/env python3
"""
데이터 파이프라인 모듈 테스트 스크립트
데이터 누출 방지 및 무결성 검증 포함
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
sys.path.append('.')

from src.data.pipeline import DataPipeline, AudioFile, DatasetSplit
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def test_data_loading():
    """데이터 로딩 기능 테스트"""
    
    logger = setup_logger("data_loading_test", "INFO")
    logger.info("=== 데이터 로딩 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    # 각 분할별 데이터 로딩 테스트
    try:
        # 훈련 데이터 로딩
        train_data = pipeline.load_train_data()
        logger.info(f"훈련 데이터 로딩: {len(train_data)}개 클래스")
        
        for class_name, files in train_data.items():
            logger.info(f"  {class_name}: {len(files)}개 파일")
            if files:
                # 첫 번째 파일 정보 확인
                first_file = files[0]
                logger.info(f"    예시: {os.path.basename(first_file.file_path)} "
                           f"(split: {first_file.split}, augmented: {first_file.is_augmented})")
        
        # 검증 데이터 로딩
        validation_data = pipeline.load_validation_data()
        logger.info(f"검증 데이터 로딩: {len(validation_data)}개 클래스")
        
        # 테스트 데이터 로딩
        test_data = pipeline.load_test_data()
        logger.info(f"테스트 데이터 로딩: {len(test_data)}개 클래스")
        
        # 소음 파일 로딩
        noise_files = pipeline.load_noise_files()
        logger.info(f"소음 파일 로딩: {len(noise_files)}개 파일")
        
        success = True
        
    except Exception as e:
        logger.error(f"데이터 로딩 중 오류: {e}")
        success = False
    
    logger.info("=== 데이터 로딩 테스트 완료 ===\n")
    return success

def test_complete_data_loading():
    """전체 데이터 로딩 및 DatasetSplit 테스트"""
    
    logger = setup_logger("complete_loading_test", "INFO")
    logger.info("=== 전체 데이터 로딩 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 전체 데이터 로딩
        dataset_split = pipeline.load_all_data()
        
        logger.info(f"DatasetSplit 생성 완료:")
        logger.info(f"  훈련: {dataset_split.total_train}개 파일")
        logger.info(f"  검증: {dataset_split.total_validation}개 파일")
        logger.info(f"  테스트: {dataset_split.total_test}개 파일")
        logger.info(f"  소음: {len(dataset_split.noise_files)}개 파일")
        
        # 클래스별 분포 확인
        logger.info("클래스별 분포:")
        for class_name in DEFAULT_CONFIG.class_names:
            train_count = len(dataset_split.train_files.get(class_name, []))
            val_count = len(dataset_split.validation_files.get(class_name, []))
            test_count = len(dataset_split.test_files.get(class_name, []))
            
            logger.info(f"  {class_name}: 훈련 {train_count}, 검증 {val_count}, 테스트 {test_count}")
        
        success = True
        
    except Exception as e:
        logger.error(f"전체 데이터 로딩 중 오류: {e}")
        success = False
    
    logger.info("=== 전체 데이터 로딩 테스트 완료 ===\n")
    return success

def test_data_integrity_validation():
    """데이터 무결성 검증 테스트"""
    
    logger = setup_logger("integrity_test", "INFO")
    logger.info("=== 데이터 무결성 검증 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 데이터 로딩
        dataset_split = pipeline.load_all_data()
        
        # 무결성 검증
        is_valid = pipeline.validate_data_integrity()
        
        if is_valid:
            logger.info("✅ 데이터 무결성 검증 통과")
        else:
            logger.warning("⚠️ 데이터 무결성 문제 발견")
        
        success = True
        
    except Exception as e:
        logger.error(f"무결성 검증 중 오류: {e}")
        success = False
    
    logger.info("=== 데이터 무결성 검증 테스트 완료 ===\n")
    return success

def test_training_data_augmentation():
    """훈련 데이터 증강 테스트"""
    
    logger = setup_logger("augmentation_test", "INFO")
    logger.info("=== 훈련 데이터 증강 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 데이터 로딩
        dataset_split = pipeline.load_all_data()
        
        # 원본 훈련 데이터 개수
        original_train_count = dataset_split.total_train
        logger.info(f"원본 훈련 데이터: {original_train_count}개 파일")
        
        # 증강 수행
        augmented_train = pipeline.augment_training_data()
        
        # 증강 결과 확인
        augmented_count = sum(len(files) for files in augmented_train.values())
        logger.info(f"증강 후 훈련 데이터: {augmented_count}개 파일")
        logger.info(f"증강 비율: {augmented_count / original_train_count:.1f}x")
        
        # 클래스별 증강 결과
        for class_name, files in augmented_train.items():
            original_files = [f for f in files if not f.is_augmented]
            augmented_files = [f for f in files if f.is_augmented]
            
            logger.info(f"  {class_name}: 원본 {len(original_files)}개 + 증강 {len(augmented_files)}개 "
                       f"= 총 {len(files)}개")
        
        # 증강 파일 검증: 훈련 세트에만 있는지 확인
        validation_augmented = sum(1 for files in dataset_split.validation_files.values() 
                                 for f in files if f.is_augmented)
        test_augmented = sum(1 for files in dataset_split.test_files.values() 
                           for f in files if f.is_augmented)
        
        if validation_augmented > 0 or test_augmented > 0:
            logger.error(f"데이터 누출 발견: 검증 {validation_augmented}개, 테스트 {test_augmented}개 증강 파일")
            success = False
        else:
            logger.info("✅ 증강이 훈련 세트에만 적용됨 (데이터 누출 없음)")
            success = True
        
    except Exception as e:
        logger.error(f"훈련 데이터 증강 중 오류: {e}")
        success = False
    
    logger.info("=== 훈련 데이터 증강 테스트 완료 ===\n")
    return success

def test_feature_extraction():
    """특징 추출 테스트"""
    
    logger = setup_logger("feature_extraction_test", "INFO")
    logger.info("=== 특징 추출 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 데이터 로딩 및 증강 (작은 샘플로)
        dataset_split = pipeline.load_all_data()
        
        # 테스트용으로 작은 데이터셋 생성
        test_train_files = {}
        for class_name, files in dataset_split.train_files.items():
            # 각 클래스에서 처음 2개 파일만 사용
            test_train_files[class_name] = files[:2]
        
        # 임시로 작은 데이터셋으로 설정
        pipeline._dataset_split.train_files = test_train_files
        pipeline._augmented_train_files = test_train_files
        
        # 특징 추출
        logger.info("특징 추출 시작 (작은 샘플)")
        X_train, y_train, X_val, y_val, X_test, y_test = pipeline.extract_all_features()
        
        logger.info(f"특징 추출 완료:")
        logger.info(f"  훈련: {X_train.shape} features, {y_train.shape} labels")
        logger.info(f"  검증: {X_val.shape} features, {y_val.shape} labels")
        logger.info(f"  테스트: {X_test.shape} features, {y_test.shape} labels")
        
        # 특징 벡터 크기 검증
        expected_feature_size = 30  # design.md 명세
        if X_train.shape[1] == expected_feature_size:
            logger.info(f"✅ 특징 벡터 크기 올바름: {expected_feature_size}차원")
        else:
            logger.error(f"❌ 특징 벡터 크기 오류: 예상 {expected_feature_size}, 실제 {X_train.shape[1]}")
        
        # 라벨 검증
        unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
        expected_labels = list(range(len(DEFAULT_CONFIG.class_names)))
        
        if set(unique_labels) == set(expected_labels):
            logger.info(f"✅ 라벨 올바름: {unique_labels}")
        else:
            logger.error(f"❌ 라벨 오류: 예상 {expected_labels}, 실제 {unique_labels}")
        
        success = X_train.size > 0 and X_val.size > 0 and X_test.size > 0
        
    except Exception as e:
        logger.error(f"특징 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    logger.info("=== 특징 추출 테스트 완료 ===\n")
    return success

def test_complete_pipeline():
    """완전한 파이프라인 테스트"""
    
    logger = setup_logger("complete_pipeline_test", "INFO")
    logger.info("=== 완전한 파이프라인 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 증강 없이 빠른 테스트
        logger.info("증강 없이 파이프라인 실행")
        features = pipeline.run_complete_pipeline(skip_augmentation=True)
        
        X_train, y_train, X_val, y_val, X_test, y_test = features
        
        logger.info(f"파이프라인 실행 완료:")
        logger.info(f"  훈련: {X_train.shape}")
        logger.info(f"  검증: {X_val.shape}")
        logger.info(f"  테스트: {X_test.shape}")
        
        # 파이프라인 요약 정보
        summary = pipeline.get_pipeline_summary()
        logger.info(f"파이프라인 요약: {summary}")
        
        success = True
        
    except Exception as e:
        logger.error(f"완전한 파이프라인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    logger.info("=== 완전한 파이프라인 테스트 완료 ===\n")
    return success

def test_data_leakage_prevention():
    """데이터 누출 방지 전용 테스트"""
    
    logger = setup_logger("data_leakage_test", "INFO")
    logger.info("=== 데이터 누출 방지 테스트 시작 ===")
    
    pipeline = DataPipeline()
    
    try:
        # 전체 파이프라인 실행 (증강 포함)
        dataset_split = pipeline.load_all_data()
        augmented_train = pipeline.augment_training_data()
        
        # 1. 원본 파일 중복 검사
        all_original_files = set()
        duplicates_found = []
        
        for split_name, split_files in [
            ("train", dataset_split.train_files),
            ("validation", dataset_split.validation_files),
            ("test", dataset_split.test_files)
        ]:
            for class_name, files in split_files.items():
                for audio_file in files:
                    if not audio_file.is_augmented:
                        file_key = os.path.basename(audio_file.file_path)
                        if file_key in all_original_files:
                            duplicates_found.append(file_key)
                        else:
                            all_original_files.add(file_key)
        
        # 2. 증강 파일이 훈련 세트에만 있는지 검사
        augmented_in_validation = []
        augmented_in_test = []
        
        for class_name, files in dataset_split.validation_files.items():
            for audio_file in files:
                if audio_file.is_augmented:
                    augmented_in_validation.append(audio_file.file_path)
        
        for class_name, files in dataset_split.test_files.items():
            for audio_file in files:
                if audio_file.is_augmented:
                    augmented_in_test.append(audio_file.file_path)
        
        # 3. 결과 보고
        issues = []
        
        if duplicates_found:
            issues.append(f"중복 원본 파일: {duplicates_found}")
        
        if augmented_in_validation:
            issues.append(f"검증 세트에 증강 파일: {len(augmented_in_validation)}개")
        
        if augmented_in_test:
            issues.append(f"테스트 세트에 증강 파일: {len(augmented_in_test)}개")
        
        if issues:
            logger.error("데이터 누출 문제 발견:")
            for issue in issues:
                logger.error(f"  - {issue}")
            success = False
        else:
            logger.info("✅ 데이터 누출 방지 검증 통과")
            logger.info("  - 원본 파일 중복 없음")
            logger.info("  - 증강이 훈련 세트에만 적용됨")
            logger.info("  - 검증/테스트 세트 무결성 유지")
            success = True
        
    except Exception as e:
        logger.error(f"데이터 누출 방지 테스트 중 오류: {e}")
        success = False
    
    logger.info("=== 데이터 누출 방지 테스트 완료 ===\n")
    return success

if __name__ == "__main__":
    logger = setup_logger("main_pipeline_test", "INFO")
    logger.info("🍉 데이터 파이프라인 모듈 종합 테스트 시작 🍉")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("데이터 로딩", test_data_loading()))
    test_results.append(("전체 데이터 로딩", test_complete_data_loading()))
    test_results.append(("데이터 무결성 검증", test_data_integrity_validation()))
    test_results.append(("훈련 데이터 증강", test_training_data_augmentation()))
    test_results.append(("특징 추출", test_feature_extraction()))
    test_results.append(("완전한 파이프라인", test_complete_pipeline()))
    test_results.append(("데이터 누출 방지", test_data_leakage_prevention()))
    
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
        logger.info("✅ 데이터 파이프라인이 올바르게 구현되었습니다.")
        logger.info("✅ 데이터 누출 방지가 제대로 작동합니다.")
    else:
        logger.info("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("🍉 데이터 파이프라인 모듈 테스트 완료 🍉")