"""
확장된 오디오 특징 추출 모듈 테스트

다양한 오디오 형식, 극단적 경우, 에러 상황을 포함한 포괄적인 테스트 스위트
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import librosa
import soundfile as sf
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.audio.feature_extraction import (
    extract_features, AudioFeatureExtractor, FeatureVector
)
from config import DEFAULT_CONFIG


class TestFeatureExtractionExtended:
    """확장된 특징 추출 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = AudioFeatureExtractor()
        
        # 다양한 형식의 테스트 오디오 데이터 생성
        self.sample_rates = [8000, 16000, 22050, 44100, 48000]
        self.durations = [0.1, 0.5, 1.0, 2.0, 5.0]  # 초 단위
        self.test_files = {}
        
        # 각 조합에 대해 테스트 파일 생성
        for sr in [22050, 44100]:  # 주요 샘플레이트만 테스트
            for duration in [0.5, 2.0]:  # 주요 길이만 테스트
                self._create_test_audio_file(sr, duration)
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_audio_file(self, sr, duration, noise_type='sine'):
        """테스트용 오디오 파일 생성"""
        n_samples = int(sr * duration)
        
        if noise_type == 'sine':
            # 사인파 생성 (440Hz)
            t = np.linspace(0, duration, n_samples)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        elif noise_type == 'white':
            # 백색 소음 생성
            audio_data = np.random.normal(0, 0.1, n_samples)
        elif noise_type == 'silence':
            # 무음 생성
            audio_data = np.zeros(n_samples)
        elif noise_type == 'impulse':
            # 충격 신호 생성
            audio_data = np.zeros(n_samples)
            audio_data[n_samples // 2] = 1.0
        else:
            # 복합 신호 (여러 주파수 조합)
            t = np.linspace(0, duration, n_samples)
            audio_data = (0.3 * np.sin(2 * np.pi * 220 * t) +
                         0.3 * np.sin(2 * np.pi * 440 * t) +
                         0.2 * np.sin(2 * np.pi * 880 * t))
        
        filename = f"test_{sr}_{duration}_{noise_type}.wav"
        filepath = os.path.join(self.temp_dir, filename)
        
        # WAV 파일로 저장
        sf.write(filepath, audio_data, sr)
        
        key = f"{sr}_{duration}_{noise_type}"
        self.test_files[key] = filepath
        
        return filepath
    
    def test_extract_features_various_sample_rates(self):
        """다양한 샘플링 레이트에 대한 특징 추출 테스트"""
        for sr in [22050, 44100]:
            filepath = self._create_test_audio_file(sr, 2.0, 'sine')
            
            features = extract_features(filepath)
            
            # 특징 벡터 검증
            assert isinstance(features, FeatureVector)
            assert len(features.mfcc) == DEFAULT_CONFIG.n_mfcc
            assert len(features.chroma) == DEFAULT_CONFIG.n_chroma
            assert not np.isnan(features.mel_mean)
            assert not np.isnan(features.mel_std)
            assert features.mel_std >= 0  # 표준편차는 음수가 될 수 없음
    
    def test_extract_features_various_durations(self):
        """다양한 길이의 오디오에 대한 특징 추출 테스트"""
        for duration in [0.1, 0.5, 1.0, 2.0]:
            if duration < 0.5:  # 너무 짧은 오디오는 건너뛰기
                continue
                
            filepath = self._create_test_audio_file(22050, duration, 'sine')
            
            features = extract_features(filepath)
            
            # 길이에 관계없이 특징 차원은 동일해야 함
            feature_array = features.to_array()
            expected_dim = DEFAULT_CONFIG.n_mfcc + 5 + DEFAULT_CONFIG.n_chroma
            assert len(feature_array) == expected_dim
            
            # 모든 특징이 유한한 값이어야 함
            assert np.all(np.isfinite(feature_array))
    
    def test_extract_features_different_signal_types(self):
        """다양한 신호 유형에 대한 특징 추출 테스트"""
        signal_types = ['sine', 'white', 'complex']
        
        for signal_type in signal_types:
            filepath = self._create_test_audio_file(22050, 2.0, signal_type)
            
            features = extract_features(filepath)
            feature_array = features.to_array()
            
            # 모든 특징이 유한해야 함
            assert np.all(np.isfinite(feature_array))
            
            # 신호 유형에 따른 특성 검증
            if signal_type == 'sine':
                # 사인파는 스펙트럴 중심이 특정 범위에 있어야 함
                assert features.spectral_centroid > 0
            elif signal_type == 'white':
                # 백색소음은 더 넓은 스펙트럴 분포를 가져야 함
                assert features.spectral_rolloff > features.spectral_centroid
    
    def test_extract_features_silence_handling(self):
        """무음 오디오 처리 테스트"""
        filepath = self._create_test_audio_file(22050, 2.0, 'silence')
        
        features = extract_features(filepath)
        feature_array = features.to_array()
        
        # 무음이어도 특징은 추출되어야 함 (0이 아닌 값들이 있을 수 있음)
        assert len(feature_array) == DEFAULT_CONFIG.n_mfcc + 5 + DEFAULT_CONFIG.n_chroma
        assert np.all(np.isfinite(feature_array))
        
        # 제로 크로싱 레이트는 무음에서 매우 낮아야 함
        assert features.zero_crossing_rate >= 0
        assert features.zero_crossing_rate < 0.1
    
    def test_extract_features_impulse_signal(self):
        """충격 신호 처리 테스트"""
        filepath = self._create_test_audio_file(22050, 2.0, 'impulse')
        
        features = extract_features(filepath)
        feature_array = features.to_array()
        
        # 충격 신호도 정상적으로 처리되어야 함
        assert np.all(np.isfinite(feature_array))
        assert features.zero_crossing_rate >= 0
    
    def test_extract_features_corrupted_file(self):
        """손상된 파일 처리 테스트"""
        # 잘못된 헤더를 가진 파일 생성
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, 'wb') as f:
            f.write(b"Not a valid audio file")
        
        # 손상된 파일에 대해서는 None이 반환되어야 함
        features = extract_features(corrupted_file)
        assert features is None
    
    def test_extract_features_nonexistent_file(self):
        """존재하지 않는 파일 처리 테스트"""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.wav")
        
        features = extract_features(nonexistent_file)
        assert features is None
    
    def test_extract_features_empty_file(self):
        """빈 파일 처리 테스트"""
        empty_file = os.path.join(self.temp_dir, "empty.wav")
        with open(empty_file, 'wb') as f:
            pass  # 빈 파일 생성
        
        features = extract_features(empty_file)
        assert features is None
    
    def test_extract_features_very_short_audio(self):
        """매우 짧은 오디오 처리 테스트"""
        # 0.01초 (매우 짧은) 오디오 생성
        filepath = self._create_test_audio_file(22050, 0.01, 'sine')
        
        # 짧은 오디오는 처리할 수 없을 수 있음
        features = extract_features(filepath)
        
        if features is not None:
            # 처리되었다면 유효한 특징이어야 함
            feature_array = features.to_array()
            assert np.all(np.isfinite(feature_array))
        # 처리되지 않았다면 None을 반환하는 것이 정상
    
    def test_extract_features_very_long_audio(self):
        """매우 긴 오디오 처리 테스트"""
        # 10초 (긴) 오디오 생성
        filepath = self._create_test_audio_file(22050, 10.0, 'sine')
        
        features = extract_features(filepath)
        
        assert features is not None
        feature_array = features.to_array()
        assert len(feature_array) == DEFAULT_CONFIG.n_mfcc + 5 + DEFAULT_CONFIG.n_chroma
        assert np.all(np.isfinite(feature_array))
    
    def test_extract_features_high_amplitude(self):
        """높은 진폭 신호 처리 테스트"""
        # 높은 진폭 신호 생성 (클리핑 가능성)
        sr = 22050
        duration = 2.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples)
        audio_data = 2.0 * np.sin(2 * np.pi * 440 * t)  # 클리핑될 수 있는 진폭
        
        filepath = os.path.join(self.temp_dir, "high_amplitude.wav")
        sf.write(filepath, audio_data, sr)
        
        features = extract_features(filepath)
        
        assert features is not None
        feature_array = features.to_array()
        assert np.all(np.isfinite(feature_array))
    
    def test_extract_features_low_amplitude(self):
        """낮은 진폭 신호 처리 테스트"""
        # 매우 낮은 진폭 신호 생성
        sr = 22050
        duration = 2.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples)
        audio_data = 0.001 * np.sin(2 * np.pi * 440 * t)  # 매우 작은 진폭
        
        filepath = os.path.join(self.temp_dir, "low_amplitude.wav")
        sf.write(filepath, audio_data, sr)
        
        features = extract_features(filepath)
        
        assert features is not None
        feature_array = features.to_array()
        assert np.all(np.isfinite(feature_array))
    
    @patch('librosa.load')
    def test_extract_features_librosa_error(self, mock_load):
        """librosa 로딩 에러 처리 테스트"""
        mock_load.side_effect = Exception("Librosa loading error")
        
        filepath = self._create_test_audio_file(22050, 2.0, 'sine')
        features = extract_features(filepath)
        
        # 에러 발생 시 None을 반환해야 함
        assert features is None
    
    def test_feature_vector_properties(self):
        """FeatureVector 클래스 속성 테스트"""
        filepath = self._create_test_audio_file(22050, 2.0, 'sine')
        features = extract_features(filepath)
        
        # feature_names 속성 테스트
        names = features.feature_names
        expected_count = DEFAULT_CONFIG.n_mfcc + 5 + DEFAULT_CONFIG.n_chroma
        assert len(names) == expected_count
        
        # 이름들이 올바른 형식인지 확인
        mfcc_count = sum(1 for name in names if name.startswith('mfcc_'))
        chroma_count = sum(1 for name in names if name.startswith('chroma_'))
        stat_count = sum(1 for name in names if name in 
                        ['mel_mean', 'mel_std', 'spectral_centroid', 
                         'spectral_rolloff', 'zero_crossing_rate'])
        
        assert mfcc_count == DEFAULT_CONFIG.n_mfcc
        assert chroma_count == DEFAULT_CONFIG.n_chroma
        assert stat_count == 5
    
    def test_feature_vector_to_array_consistency(self):
        """FeatureVector.to_array() 일관성 테스트"""
        filepath = self._create_test_audio_file(22050, 2.0, 'sine')
        features = extract_features(filepath)
        
        # 여러 번 호출해도 같은 결과가 나와야 함
        array1 = features.to_array()
        array2 = features.to_array()
        
        np.testing.assert_array_equal(array1, array2)
        
        # 예상된 구조인지 확인
        expected_structure = np.concatenate([
            features.mfcc,
            [features.mel_mean, features.mel_std, features.spectral_centroid,
             features.spectral_rolloff, features.zero_crossing_rate],
            features.chroma
        ])
        
        np.testing.assert_array_equal(array1, expected_structure)
    
    def test_audio_feature_extractor_initialization(self):
        """AudioFeatureExtractor 초기화 테스트"""
        # 기본 초기화
        extractor = AudioFeatureExtractor()
        assert extractor.sr == DEFAULT_CONFIG.sr
        assert extractor.hop_length == DEFAULT_CONFIG.hop_length
        
        # 사용자 정의 초기화
        custom_extractor = AudioFeatureExtractor(sr=44100, hop_length=1024)
        assert custom_extractor.sr == 44100
        assert custom_extractor.hop_length == 1024
    
    def test_extract_features_batch_processing(self):
        """배치 처리 테스트"""
        # 여러 파일 생성
        test_files = []
        for i in range(3):
            filepath = self._create_test_audio_file(22050, 1.0, 'sine')
            new_path = filepath.replace('.wav', f'_{i}.wav')
            os.rename(filepath, new_path)
            test_files.append(new_path)
        
        # 각 파일에 대해 특징 추출
        all_features = []
        for filepath in test_files:
            features = extract_features(filepath)
            assert features is not None
            all_features.append(features.to_array())
        
        # 모든 특징 벡터가 같은 차원을 가져야 함
        dimensions = [len(features) for features in all_features]
        assert len(set(dimensions)) == 1  # 모든 차원이 동일해야 함
    
    def test_memory_usage_with_large_files(self):
        """큰 파일 처리 시 메모리 사용량 테스트"""
        # 상대적으로 큰 오디오 파일 생성 (30초)
        filepath = self._create_test_audio_file(22050, 30.0, 'sine')
        
        # 메모리 사용량 모니터링은 실제 구현에서는 더 정교해야 함
        features = extract_features(filepath)
        
        assert features is not None
        feature_array = features.to_array()
        assert np.all(np.isfinite(feature_array))
        
        # 메모리가 적절히 해제되었는지 확인 (간접적 테스트)
        del features
        del feature_array
    
    def test_reproducibility(self):
        """재현성 테스트"""
        filepath = self._create_test_audio_file(22050, 2.0, 'sine')
        
        # 같은 파일에 대해 여러 번 특징 추출
        features1 = extract_features(filepath)
        features2 = extract_features(filepath)
        
        # 결과가 동일해야 함
        array1 = features1.to_array()
        array2 = features2.to_array()
        
        np.testing.assert_array_almost_equal(array1, array2, decimal=10)
    
    def test_different_audio_formats(self):
        """다양한 오디오 형식 테스트 (WAV만 지원하는 경우)"""
        # 현재는 WAV만 테스트하지만, 향후 다른 형식 지원 시 확장 가능
        formats = ['.wav']
        
        for fmt in formats:
            if fmt == '.wav':
                filepath = self._create_test_audio_file(22050, 2.0, 'sine')
                features = extract_features(filepath)
                assert features is not None
        
        # 지원하지 않는 형식 테스트
        unsupported_file = os.path.join(self.temp_dir, "test.mp3")
        with open(unsupported_file, 'w') as f:
            f.write("fake mp3 content")
        
        # MP3는 librosa가 처리할 수 있지만, 실제로는 유효하지 않은 파일이므로 None 반환
        features = extract_features(unsupported_file)
        # 결과는 구현에 따라 다를 수 있음


# 성능 테스트
class TestFeatureExtractionPerformance:
    """특징 추출 성능 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        import time
        
        # 테스트 파일 생성
        sr = 22050
        duration = 5.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        filepath = os.path.join(self.temp_dir, "benchmark.wav")
        sf.write(filepath, audio_data, sr)
        
        # 성능 측정
        start_time = time.time()
        features = extract_features(filepath)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert features is not None
        
        # 5초 오디오 처리가 1초 이내에 완료되어야 함 (기준은 조정 가능)
        assert processing_time < 1.0, f"Processing took {processing_time:.2f}s, expected < 1.0s"
        
        # 처리 속도 로깅 (실제 구현에서는 로거 사용)
        print(f"Feature extraction took {processing_time:.3f}s for {duration}s audio")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])