#!/usr/bin/env python3
"""
데이터 증강 모듈 테스트 스크립트
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
sys.path.append('.')

from src.data.augmentation import augment_noise, AudioAugmentor, BatchAugmentor
from src.utils.logger import setup_logger
from config import DEFAULT_CONFIG

def test_snr_calculation():
    """SNR 계산 및 소음 스케일링 테스트"""
    
    logger = setup_logger("snr_test", "INFO")
    logger.info("=== SNR 계산 테스트 시작 ===")
    
    augmentor = AudioAugmentor()
    
    # 테스트 신호 생성
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 440Hz 사인파
    noise = np.random.normal(0, 0.1, 22050)  # 가우시안 노이즈
    
    # RMS 계산 테스트
    signal_rms = augmentor.calculate_rms(signal)
    noise_rms = augmentor.calculate_rms(noise)
    
    logger.info(f"신호 RMS: {signal_rms:.4f}")
    logger.info(f"소음 RMS: {noise_rms:.4f}")
    
    # SNR 계산 테스트
    original_snr = augmentor.calculate_snr(signal, noise)
    logger.info(f"원본 SNR: {original_snr:.2f} dB")
    
    # 다양한 목표 SNR에 대해 스케일링 테스트
    target_snrs = [-5, 0, 5, 10]
    
    for target_snr in target_snrs:
        scaled_noise = augmentor.scale_noise_for_snr(signal, noise, target_snr)
        actual_snr = augmentor.calculate_snr(signal, scaled_noise)
        
        logger.info(f"목표 SNR: {target_snr:+.0f}dB → 실제 SNR: {actual_snr:+.2f}dB "
                   f"(차이: {abs(target_snr - actual_snr):.2f}dB)")
        
        # 혼합 테스트
        mixed = augmentor.mix_audio_with_noise(signal, scaled_noise, target_snr)
        mixed_snr = augmentor.calculate_snr(signal, scaled_noise)
        
        # 품질 검증
        is_valid = augmentor.validate_augmented_audio(mixed, signal)
        logger.info(f"  혼합 품질: {'✓' if is_valid else '✗'}")
    
    logger.info("=== SNR 계산 테스트 완료 ===\n")
    return True

def test_audio_augmentation_with_real_files():
    """실제 오디오 파일을 사용한 증강 테스트"""
    
    logger = setup_logger("augmentation_test", "INFO")
    logger.info("=== 실제 파일 증강 테스트 시작 ===")
    
    # 테스트할 오디오 파일 찾기
    test_files = []
    train_dirs = [
        "data/raw/train/watermelon_A",
        "data/raw/train/watermelon_B", 
        "data/raw/train/watermelon_C"
    ]
    
    for train_dir in train_dirs:
        if os.path.exists(train_dir):
            files = [f for f in os.listdir(train_dir) if f.lower().endswith('.wav')]
            if files:
                test_files.append(os.path.join(train_dir, files[0]))  # 첫 번째 파일만
                logger.info(f"테스트 파일: {files[0]} from {train_dir}")
                break
    
    if not test_files:
        logger.warning("테스트할 오디오 파일을 찾을 수 없습니다.")
        return False
    
    # 소음 파일 찾기
    config = DEFAULT_CONFIG
    noise_files = config.get_all_noise_files()
    
    if not noise_files:
        logger.warning("소음 파일을 찾을 수 없습니다. 가상 소음으로 테스트합니다.")
        # 임시 소음 파일 생성
        temp_dir = tempfile.mkdtemp()
        temp_noise = os.path.join(temp_dir, "test_noise.wav")
        
        # 간단한 노이즈 생성
        import soundfile as sf
        noise_data = np.random.normal(0, 0.1, 22050)
        sf.write(temp_noise, noise_data, 22050)
        noise_files = [temp_noise]
    else:
        temp_dir = None
        logger.info(f"발견된 소음 파일: {len(noise_files)}개")
        for noise_file in noise_files[:3]:  # 처음 3개만 출력
            logger.info(f"  - {os.path.basename(noise_file)}")
    
    # 임시 출력 디렉토리 생성
    output_dir = tempfile.mkdtemp()
    logger.info(f"출력 디렉토리: {output_dir}")
    
    try:
        # 증강 수행
        test_file = test_files[0]
        snr_levels = [-5, 0, 5]  # 테스트용 축소된 SNR 레벨
        selected_noise = noise_files[:2]  # 처음 2개 소음 파일만 사용
        
        logger.info(f"증강 시작: {os.path.basename(test_file)}")
        logger.info(f"SNR 레벨: {snr_levels}")
        logger.info(f"소음 파일: {len(selected_noise)}개")
        
        augmented_files = augment_noise(
            test_file, selected_noise, snr_levels, output_dir, config
        )
        
        logger.info(f"증강 완료: {len(augmented_files)}개 파일 생성")
        
        # 생성된 파일 검증
        for augmented_file in augmented_files:
            if os.path.exists(augmented_file):
                file_size = os.path.getsize(augmented_file)
                logger.info(f"  ✓ {os.path.basename(augmented_file)} ({file_size} bytes)")
            else:
                logger.error(f"  ✗ {os.path.basename(augmented_file)} (파일 없음)")
        
        success = len(augmented_files) > 0
        
    finally:
        # 정리
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    logger.info("=== 실제 파일 증강 테스트 완료 ===\n")
    return success

def test_batch_augmentation():
    """배치 증강 테스트"""
    
    logger = setup_logger("batch_test", "INFO")
    logger.info("=== 배치 증강 테스트 시작 ===")
    
    batch_augmentor = BatchAugmentor()
    
    # 동적 증강 계산 테스트
    test_cases = [0, 1, 2, 4, 8, 12]
    
    for noise_count in test_cases:
        snr_levels, actual_factor = batch_augmentor.calculate_dynamic_augmentation(noise_count)
        logger.info(f"소음 파일 {noise_count}개 → SNR 레벨: {len(snr_levels)}개, "
                   f"증강 배수: {actual_factor}")
    
    # 소음 타입 추출 테스트
    test_paths = [
        "/path/to/noise/environmental/retail/homeplus/ambient.wav",
        "/path/to/noise/environmental/retail/emart/crowd.wav", 
        "/path/to/noise/mechanical/fan.wav",
        "/path/to/noise/background/office.wav",
        "/unknown/path/mystery.wav"
    ]
    
    for test_path in test_paths:
        noise_type = batch_augmentor._extract_noise_type(test_path)
        logger.info(f"경로: {test_path} → 타입: {noise_type}")
    
    logger.info("=== 배치 증강 테스트 완료 ===\n")
    return True

def test_edge_cases():
    """경계 사례 테스트"""
    
    logger = setup_logger("edge_case_test", "INFO")
    logger.info("=== 경계 사례 테스트 시작 ===")
    
    augmentor = AudioAugmentor()
    
    # 1. 무음 신호 테스트
    silent_signal = np.zeros(1000)
    noise = np.random.normal(0, 0.1, 1000)
    
    snr = augmentor.calculate_snr(silent_signal, noise)
    logger.info(f"무음 신호 SNR: {snr}")
    
    # 2. 무음 소음 테스트
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))
    silent_noise = np.zeros(1000)
    
    snr = augmentor.calculate_snr(signal, silent_noise)
    logger.info(f"무음 소음 SNR: {snr}")
    
    # 3. 매우 큰 신호 테스트
    large_signal = np.ones(1000) * 100
    normal_noise = np.random.normal(0, 0.1, 1000)
    
    mixed = augmentor.mix_audio_with_noise(large_signal, normal_noise, 0)
    is_valid = augmentor.validate_augmented_audio(mixed, large_signal)
    logger.info(f"큰 신호 품질 검증: {'✓' if is_valid else '✗'}")
    logger.info(f"혼합 후 최대값: {np.max(np.abs(mixed)):.3f}")
    
    # 4. 길이가 다른 오디오 테스트
    short_audio = np.random.normal(0, 0.1, 100)
    long_target = 1000
    
    matched_audio = augmentor.load_and_match_length("dummy", long_target, 22050)
    # 실제 파일 로드는 실패하지만 길이 매칭 로직 확인용
    
    logger.info("=== 경계 사례 테스트 완료 ===\n")
    return True

if __name__ == "__main__":
    logger = setup_logger("main_test", "INFO")
    logger.info("🎵 데이터 증강 모듈 종합 테스트 시작 🎵")
    
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("SNR 계산", test_snr_calculation()))
    test_results.append(("실제 파일 증강", test_audio_augmentation_with_real_files()))
    test_results.append(("배치 증강", test_batch_augmentation()))
    test_results.append(("경계 사례", test_edge_cases()))
    
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
    else:
        logger.info("⚠️  일부 테스트가 실패했습니다. 로그를 확인해주세요.")
    
    logger.info("🎵 데이터 증강 모듈 테스트 완료 🎵")