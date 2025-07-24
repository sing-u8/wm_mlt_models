#!/usr/bin/env python3
"""
오디오 특징 추출 모듈 테스트 스크립트
"""

import sys
import os
import numpy as np
from pathlib import Path
sys.path.append('.')

from src.audio.feature_extraction import extract_features, AudioFeatureExtractor
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def test_feature_extraction():
    """특징 추출 기능을 실제 오디오 파일로 테스트합니다."""
    
    # 로거 설정
    logger = setup_logger("feature_test", "INFO")
    logger.info("=" * 60)
    logger.info("오디오 특징 추출 모듈 테스트 시작")
    
    # Config 확인
    config = DEFAULT_CONFIG
    logger.info(f"설정: SR={config.sample_rate}, MFCC={config.n_mfcc}, Chroma={config.n_chroma}")
    
    # 테스트할 오디오 파일 찾기
    test_dirs = [
        "data/raw/train/watermelon_A",
        "data/raw/train/watermelon_B", 
        "data/raw/train/watermelon_C"
    ]
    
    test_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.lower().endswith('.wav')]
            if files:
                # 각 클래스에서 첫 번째 파일만 테스트
                test_files.append(os.path.join(test_dir, files[0]))
                logger.info(f"테스트 파일 추가: {files[0]} from {test_dir}")
    
    if not test_files:
        logger.error("테스트할 오디오 파일을 찾을 수 없습니다!")
        return False
    
    logger.info(f"총 {len(test_files)}개 파일 테스트 예정")
    
    # 각 파일에 대해 특징 추출 테스트
    successful_extractions = 0
    feature_stats = []
    
    for i, audio_file in enumerate(test_files, 1):
        logger.info("=" * 50)
        logger.info(f"테스트 {i}/{len(test_files)}: {os.path.basename(audio_file)}")
        
        try:
            # 특징 추출
            feature_vector = extract_features(audio_file, config)
            
            if feature_vector is None:
                logger.error(f"특징 추출 실패: {audio_file}")
                continue
            
            # 특징 분석
            feature_array = feature_vector.to_array()
            feature_names = feature_vector.feature_names
            
            logger.info(f"✓ 특징 추출 성공!")
            logger.info(f"  - 특징 벡터 크기: {len(feature_array)}")
            logger.info(f"  - 특징 이름 수: {len(feature_names)}")
            logger.info(f"  - 평균값: {np.mean(feature_array):.4f}")
            logger.info(f"  - 표준편차: {np.std(feature_array):.4f}")
            logger.info(f"  - 최솟값: {np.min(feature_array):.4f}")
            logger.info(f"  - 최댓값: {np.max(feature_array):.4f}")
            
            # 개별 특징 확인
            logger.info("개별 특징 분석:")
            logger.info(f"  - MFCC: {feature_vector.mfcc.shape} (평균: {np.mean(feature_vector.mfcc):.4f})")
            logger.info(f"  - Mel 평균: {feature_vector.mel_mean:.4f}")
            logger.info(f"  - Mel 표준편차: {feature_vector.mel_std:.4f}")
            logger.info(f"  - Spectral Centroid: {feature_vector.spectral_centroid:.4f}")
            logger.info(f"  - Spectral Rolloff: {feature_vector.spectral_rolloff:.4f}")
            logger.info(f"  - Zero Crossing Rate: {feature_vector.zero_crossing_rate:.4f}")
            logger.info(f"  - Chroma: {feature_vector.chroma.shape} (평균: {np.mean(feature_vector.chroma):.4f})")
            
            # 유효성 검사
            if len(feature_array) != 30:  # 13 + 5 + 12 = 30
                logger.warning(f"예상 특징 크기(30)와 다름: {len(feature_array)}")
            
            if not np.isfinite(feature_array).all():
                logger.warning("특징 벡터에 NaN 또는 무한대 값 포함")
            
            feature_stats.append({
                'file': os.path.basename(audio_file),
                'feature_vector': feature_vector,
                'feature_array': feature_array
            })
            
            successful_extractions += 1
            
        except Exception as e:
            logger.error(f"테스트 중 오류 발생 {audio_file}: {e}")
            continue
    
    # 전체 결과 요약
    logger.info("=" * 60)
    logger.info("테스트 결과 요약")
    logger.info(f"성공한 특징 추출: {successful_extractions}/{len(test_files)}")
    
    if successful_extractions > 0:
        # 특징 통계 분석
        all_features = np.array([stat['feature_array'] for stat in feature_stats])
        
        logger.info(f"전체 특징 통계:")
        logger.info(f"  - 평균: {np.mean(all_features, axis=0)[:5]}... (처음 5개 특징)")
        logger.info(f"  - 표준편차: {np.std(all_features, axis=0)[:5]}... (처음 5개 특징)")
        
        # 클래스별 차이 확인 (파일이 여러 클래스에서 왔다면)
        if len(feature_stats) >= 2:
            logger.info("클래스 간 특징 차이 분석:")
            for i in range(min(3, len(feature_stats))):
                class_name = feature_stats[i]['file'].split('_')[0] if '_' in feature_stats[i]['file'] else f"파일{i+1}"
                mean_val = np.mean(feature_stats[i]['feature_array'])
                logger.info(f"  - {class_name}: 평균 특징값 {mean_val:.4f}")
    
    # 특징 이름 출력
    if successful_extractions > 0:
        feature_names = feature_stats[0]['feature_vector'].feature_names
        logger.info(f"특징 이름 목록 ({len(feature_names)}개):")
        for i, name in enumerate(feature_names):
            if i < 10 or i >= len(feature_names) - 5:  # 처음 10개와 마지막 5개만 표시
                logger.info(f"  {i+1:2d}. {name}")
            elif i == 10:
                logger.info("  ... (중간 특징들 생략)")
    
    logger.info("=" * 60)
    success_rate = successful_extractions / len(test_files) * 100
    logger.info(f"특징 추출 테스트 완료! 성공률: {success_rate:.1f}%")
    
    return successful_extractions > 0

def test_audio_validation():
    """오디오 파일 검증 기능을 테스트합니다."""
    
    logger = setup_logger("validation_test", "INFO")
    logger.info("=" * 50)
    logger.info("오디오 파일 검증 테스트 시작")
    
    extractor = AudioFeatureExtractor()
    
    # 존재하는 파일 테스트
    test_files = []
    for root, dirs, files in os.walk("data/raw/train"):
        for file in files[:2]:  # 각 디렉토리에서 처음 2개만
            if file.lower().endswith('.wav'):
                test_files.append(os.path.join(root, file))
    
    if test_files:
        logger.info(f"실제 파일 검증 테스트 ({len(test_files)}개):")
        for test_file in test_files:
            result = extractor.validate_audio_file(test_file)
            status = "✓" if result else "✗"
            logger.info(f"  {status} {os.path.basename(test_file)}")
    
    # 존재하지 않는 파일 테스트
    logger.info("존재하지 않는 파일 테스트:")
    fake_file = "non_existent_file.wav"
    result = extractor.validate_audio_file(fake_file)
    status = "✓" if not result else "✗"  # 실패해야 성공
    logger.info(f"  {status} {fake_file} (실패 예상)")
    
    logger.info("오디오 파일 검증 테스트 완료!")

if __name__ == "__main__":
    # 특징 추출 테스트
    extraction_success = test_feature_extraction()
    
    print()  # 구분선
    
    # 검증 테스트
    test_audio_validation()
    
    if extraction_success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")