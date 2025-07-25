#!/usr/bin/env python3
"""
빠른 시스템 테스트 - 패키지 호환성 문제 없이 기본 기능만 테스트
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """기본 구성 테스트"""
    print("=== 구성 테스트 ===")
    try:
        from config import DEFAULT_CONFIG
        print(f"✅ 구성 로딩 성공")
        print(f"  클래스: {DEFAULT_CONFIG.class_names}")
        print(f"  샘플 레이트: {DEFAULT_CONFIG.sample_rate}")
        print(f"  데이터 디렉토리: {DEFAULT_CONFIG.data_root_dir}")
        return True
    except Exception as e:
        print(f"❌ 구성 로딩 실패: {e}")
        return False

def test_data_availability():
    """데이터 가용성 테스트"""
    print("\n=== 데이터 가용성 테스트 ===")
    try:
        from config import DEFAULT_CONFIG
        
        total_files = 0
        for split in ['train', 'validation', 'test']:
            split_dir = Path(DEFAULT_CONFIG.raw_data_dir) / split
            if split_dir.exists():
                print(f"\n{split.upper()} 데이터:")
                split_total = 0
                for class_name in DEFAULT_CONFIG.class_names:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        wav_files = list(class_dir.glob('*.wav'))
                        split_total += len(wav_files)
                        print(f"  {class_name}: {len(wav_files)}개 파일")
                    else:
                        print(f"  {class_name}: 디렉토리 없음")
                total_files += split_total
                print(f"  {split} 소계: {split_total}개")
        
        print(f"\n총 오디오 파일: {total_files}개")
        
        # 노이즈 파일 확인
        noise_dir = Path(DEFAULT_CONFIG.noise_dir)
        if noise_dir.exists():
            noise_files = list(noise_dir.rglob('*.wav'))
            print(f"노이즈 파일: {len(noise_files)}개")
        
        return total_files > 0
        
    except Exception as e:
        print(f"❌ 데이터 확인 실패: {e}")
        return False

def test_feature_extraction():
    """특징 추출 테스트 (패키지 호환성 문제가 있을 수 있음)"""
    print("\n=== 특징 추출 테스트 ===")
    try:
        # 핵심 패키지들만 임포트 시도
        import numpy as np
        print("✅ numpy 임포트 성공")
        
        try:
            import librosa
            print("✅ librosa 임포트 성공")
            
            # 간단한 테스트 오디오 생성
            test_audio = np.random.randn(22050)  # 1초짜리 랜덤 오디오
            
            # 기본 특징 추출 테스트
            mfccs = librosa.feature.mfcc(y=test_audio, sr=22050, n_mfcc=13)
            print(f"✅ MFCC 특징 추출 성공: {mfccs.shape}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ librosa 테스트 실패: {e}")
            print("  오디오 처리는 가능하지만 특징 추출에 문제가 있을 수 있습니다.")
            return False
        
    except Exception as e:
        print(f"❌ 특징 추출 테스트 실패: {e}")
        return False

def test_directory_structure():
    """필요한 디렉토리 구조 확인 및 생성"""
    print("\n=== 디렉토리 구조 테스트 ===")
    try:
        from config import DEFAULT_CONFIG
        
        required_dirs = [
            DEFAULT_CONFIG.model_output_dir,
            "results",
            "logs",
            "checkpoints"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                print(f"✅ {dir_path}: 존재")
            else:
                path.mkdir(parents=True, exist_ok=True)
                print(f"📁 {dir_path}: 생성됨")
        
        return True
        
    except Exception as e:
        print(f"❌ 디렉토리 구조 테스트 실패: {e}")
        return False

def suggest_next_steps(test_results):
    """다음 단계 제안"""
    print("\n" + "="*60)
    print("📋 시스템 진단 결과 및 권장사항")
    print("="*60)
    
    if all(test_results.values()):
        print("🎉 모든 기본 테스트 통과! 시스템이 실행 준비되었습니다.")
        print("\n🚀 다음 단계:")
        print("1. 의존성 패키지 문제 해결:")
        print("   pip3 install --force-reinstall numpy scipy scikit-learn matplotlib librosa soundfile")
        print("\n2. 또는 간소화된 실행:")
        print("   python3 simple_pipeline.py")
        print("\n3. 또는 부분적 실행:")
        print("   python3 main.py --dry-run  # 설정만 확인")
        
    else:
        print("⚠️ 일부 문제가 발견되었습니다:")
        
        if not test_results['config']:
            print("- 구성 파일 문제: config.py를 확인하세요")
            
        if not test_results['data']:
            print("- 데이터 문제: 오디오 파일이 올바른 위치에 있는지 확인하세요")
            
        if not test_results['feature_extraction']:
            print("- 패키지 호환성 문제: 의존성 패키지를 재설치하세요")
            print("  pip3 install --force-reinstall --no-cache-dir numpy scipy scikit-learn")
            
        print("\n💡 문제 해결 후 다시 실행해보세요:")
        print("   python3 quick_test.py")

def main():
    """메인 테스트 함수"""
    print("🔍 수박 소리 분류 시스템 빠른 진단")
    print("=" * 60)
    
    # 각 테스트 실행
    test_results = {
        'config': test_config(),
        'data': test_data_availability(),
        'directories': test_directory_structure(),
        'feature_extraction': test_feature_extraction()
    }
    
    # 결과 제안
    suggest_next_steps(test_results)
    
    return all(test_results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)