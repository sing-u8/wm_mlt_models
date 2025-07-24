"""
설정 모듈에 대한 단위 테스트

이 모듈은 config/config.py의 설정 관리 시스템을 테스트합니다.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import Config, DEFAULT_CONFIG


class TestConfig:
    """Config 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config_initialization(self):
        """기본 설정으로 Config 초기화 테스트"""
        config = Config()
        
        # 기본값 확인
        assert config.sr == 22050
        assert config.hop_length == 512
        assert config.n_mfcc == 13
        assert config.n_chroma == 12
        assert len(config.snr_levels) > 0
        assert len(config.classes) == 3
    
    def test_custom_config_initialization(self):
        """사용자 정의 설정으로 Config 초기화 테스트"""
        custom_config = {
            "sr": 44100,
            "hop_length": 1024,
            "n_mfcc": 20,
            "classes": ["class_A", "class_B"]
        }
        
        config = Config(**custom_config)
        
        assert config.sr == 44100
        assert config.hop_length == 1024
        assert config.n_mfcc == 20
        assert config.classes == ["class_A", "class_B"]
        # 기본값은 유지되어야 함
        assert config.n_chroma == 12
    
    def test_config_validation_valid(self):
        """유효한 설정 검증 테스트"""
        config = Config()
        
        # 검증이 성공해야 함 (예외가 발생하지 않아야 함)
        try:
            config.validate()
        except Exception as e:
            pytest.fail(f"Valid config validation failed: {e}")
    
    def test_config_validation_invalid_sr(self):
        """잘못된 샘플링 레이트 검증 테스트"""
        with pytest.raises(ValueError, match="sr must be positive"):
            Config(sr=-1000)
    
    def test_config_validation_invalid_hop_length(self):
        """잘못된 hop_length 검증 테스트"""
        with pytest.raises(ValueError, match="hop_length must be positive"):
            Config(hop_length=0)
    
    def test_config_validation_invalid_mfcc(self):
        """잘못된 MFCC 계수 개수 검증 테스트"""
        with pytest.raises(ValueError, match="n_mfcc must be positive"):
            Config(n_mfcc=-5)
    
    def test_config_validation_invalid_chroma(self):
        """잘못된 크로마 특징 개수 검증 테스트"""
        with pytest.raises(ValueError, match="n_chroma must be positive"):
            Config(n_chroma=0)
    
    def test_config_validation_empty_classes(self):
        """빈 클래스 목록 검증 테스트"""
        with pytest.raises(ValueError, match="classes cannot be empty"):
            Config(classes=[])
    
    def test_config_validation_empty_snr_levels(self):
        """빈 SNR 레벨 목록 검증 테스트"""
        with pytest.raises(ValueError, match="snr_levels cannot be empty"):
            Config(snr_levels=[])
    
    def test_config_save_and_load(self):
        """설정 저장 및 로딩 테스트"""
        config = Config(sr=44100, n_mfcc=20)
        
        # 저장 테스트
        config.save(self.test_config_file)
        assert os.path.exists(self.test_config_file)
        
        # 로딩 테스트
        loaded_config = Config.load(self.test_config_file)
        
        assert loaded_config.sr == 44100
        assert loaded_config.n_mfcc == 20
        assert loaded_config.hop_length == config.hop_length  # 기본값 유지
    
    def test_config_load_nonexistent_file(self):
        """존재하지 않는 파일 로딩 테스트"""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent_config.json")
    
    def test_config_load_invalid_json(self):
        """잘못된 JSON 파일 로딩 테스트"""
        # 잘못된 JSON 파일 생성
        with open(self.test_config_file, 'w') as f:
            f.write("{ invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            Config.load(self.test_config_file)
    
    def test_config_to_dict(self):
        """설정을 딕셔너리로 변환 테스트"""
        config = Config(sr=44100, n_mfcc=20)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["sr"] == 44100
        assert config_dict["n_mfcc"] == 20
        assert "hop_length" in config_dict
        assert "classes" in config_dict
    
    def test_config_from_dict(self):
        """딕셔너리에서 설정 생성 테스트"""
        config_dict = {
            "sr": 48000,
            "hop_length": 256,
            "n_mfcc": 15,
            "classes": ["A", "B", "C", "D"]
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.sr == 48000
        assert config.hop_length == 256
        assert config.n_mfcc == 15
        assert config.classes == ["A", "B", "C", "D"]
    
    def test_config_update(self):
        """설정 업데이트 테스트"""
        config = Config()
        original_sr = config.sr
        
        updates = {"sr": 48000, "n_mfcc": 25}
        config.update(updates)
        
        assert config.sr == 48000
        assert config.n_mfcc == 25
        assert config.sr != original_sr
    
    def test_config_reset_to_defaults(self):
        """기본값으로 재설정 테스트"""
        config = Config(sr=48000, n_mfcc=25)
        
        # 값이 변경되었는지 확인
        assert config.sr != DEFAULT_CONFIG["sr"]
        assert config.n_mfcc != DEFAULT_CONFIG["n_mfcc"]
        
        # 기본값으로 재설정
        config.reset_to_defaults()
        
        assert config.sr == DEFAULT_CONFIG["sr"]
        assert config.n_mfcc == DEFAULT_CONFIG["n_mfcc"]
    
    def test_config_get_feature_dimension(self):
        """특징 차원 계산 테스트"""
        config = Config(n_mfcc=13, n_chroma=12)
        
        # MFCC(13) + 통계(5) + Chroma(12) = 30
        expected_dim = 13 + 5 + 12
        assert config.get_feature_dimension() == expected_dim
    
    def test_config_get_noise_files_recursive(self):
        """재귀적 소음 파일 검색 테스트"""
        # 테스트 소음 디렉토리 구조 생성
        noise_dir = os.path.join(self.temp_dir, "noise")
        retail_dir = os.path.join(noise_dir, "environmental", "retail")
        homeplus_dir = os.path.join(retail_dir, "homeplus")
        emart_dir = os.path.join(retail_dir, "emart")
        
        os.makedirs(homeplus_dir)
        os.makedirs(emart_dir)
        
        # 테스트 소음 파일들 생성
        noise_files = [
            os.path.join(homeplus_dir, "noise1.wav"),
            os.path.join(homeplus_dir, "noise2.wav"),
            os.path.join(emart_dir, "noise3.wav"),
            os.path.join(emart_dir, "noise4.wav"),
            os.path.join(noise_dir, "ignore.txt")  # 비오디오 파일
        ]
        
        for filepath in noise_files:
            with open(filepath, 'w') as f:
                f.write("fake audio data")
        
        config = Config(noise_directory=noise_dir)
        found_files = config.get_noise_files_recursive()
        
        # 4개의 wav 파일만 찾아져야 함
        assert len(found_files) == 4
        assert all(f.endswith('.wav') for f in found_files)
    
    def test_config_validate_paths(self):
        """경로 검증 테스트"""
        # 존재하는 디렉토리로 설정
        config = Config(
            data_directory=self.temp_dir,
            noise_directory=self.temp_dir
        )
        
        # 검증이 성공해야 함
        try:
            config.validate_paths()
        except Exception as e:
            pytest.fail(f"Path validation failed: {e}")
    
    def test_config_validate_paths_nonexistent(self):
        """존재하지 않는 경로 검증 테스트"""
        config = Config(data_directory="/nonexistent/path")
        
        with pytest.raises(ValueError, match="data_directory does not exist"):
            config.validate_paths()
    
    def test_config_copy(self):
        """설정 복사 테스트"""
        original = Config(sr=44100, n_mfcc=20)
        copied = original.copy()
        
        # 값이 같아야 함
        assert copied.sr == original.sr
        assert copied.n_mfcc == original.n_mfcc
        
        # 독립적인 객체여야 함
        copied.sr = 48000
        assert original.sr == 44100  # 원본은 변경되지 않아야 함
    
    def test_config_merge(self):
        """설정 병합 테스트"""
        config1 = Config(sr=44100, n_mfcc=20)
        config2 = Config(hop_length=256, n_chroma=24)
        
        merged = config1.merge(config2)
        
        # config1의 값들이 유지되어야 함
        assert merged.sr == 44100
        assert merged.n_mfcc == 20
        
        # config2의 값들이 추가되어야 함
        assert merged.hop_length == 256
        assert merged.n_chroma == 24
    
    def test_config_environment_variables(self):
        """환경 변수에서 설정 로딩 테스트"""
        with patch.dict(os.environ, {
            'WM_SR': '48000',
            'WM_HOP_LENGTH': '1024',
            'WM_N_MFCC': '15'
        }):
            config = Config.from_environment()
            
            assert config.sr == 48000
            assert config.hop_length == 1024
            assert config.n_mfcc == 15


class TestDefaultConfig:
    """DEFAULT_CONFIG 상수 테스트"""
    
    def test_default_config_completeness(self):
        """기본 설정 완성도 테스트"""
        required_keys = [
            'sr', 'hop_length', 'n_mfcc', 'n_chroma',
            'classes', 'snr_levels', 'data_directory',
            'noise_directory', 'augmentation_factor'
        ]
        
        for key in required_keys:
            assert key in DEFAULT_CONFIG, f"Missing required key: {key}"
    
    def test_default_config_values(self):
        """기본 설정 값 유효성 테스트"""
        # 양수 값들 확인
        assert DEFAULT_CONFIG['sr'] > 0
        assert DEFAULT_CONFIG['hop_length'] > 0
        assert DEFAULT_CONFIG['n_mfcc'] > 0
        assert DEFAULT_CONFIG['n_chroma'] > 0
        assert DEFAULT_CONFIG['augmentation_factor'] > 0
        
        # 비어있지 않은 목록들 확인
        assert len(DEFAULT_CONFIG['classes']) > 0
        assert len(DEFAULT_CONFIG['snr_levels']) > 0
        
        # 클래스 이름이 문자열인지 확인
        assert all(isinstance(cls, str) for cls in DEFAULT_CONFIG['classes'])
        
        # SNR 레벨이 숫자인지 확인
        assert all(isinstance(snr, (int, float)) for snr in DEFAULT_CONFIG['snr_levels'])
    
    def test_default_config_immutability(self):
        """기본 설정 불변성 테스트"""
        original_sr = DEFAULT_CONFIG['sr']
        
        # Config 객체 생성해도 원본이 변경되지 않아야 함
        config = Config(sr=48000)
        assert DEFAULT_CONFIG['sr'] == original_sr
        
        # 직접 수정 시도 (실제로는 권장하지 않음)
        try:
            DEFAULT_CONFIG['sr'] = 48000
            # 수정 후 다시 원복
            DEFAULT_CONFIG['sr'] = original_sr
        except Exception:
            # 불변 객체라면 예외가 발생할 수 있음
            pass


# 통합 테스트
class TestConfigIntegration:
    """Config 통합 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # 테스트 데이터 디렉토리 구조 생성
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.noise_dir = os.path.join(self.temp_dir, "noise")
        os.makedirs(self.data_dir)
        os.makedirs(self.noise_dir)
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_config_workflow(self):
        """전체 설정 워크플로 테스트"""
        # 1. 설정 생성
        config = Config(
            sr=44100,
            n_mfcc=15,
            data_directory=self.data_dir,
            noise_directory=self.noise_dir
        )
        
        # 2. 검증
        config.validate()
        config.validate_paths()
        
        # 3. 저장
        config.save(self.config_file)
        
        # 4. 로딩
        loaded_config = Config.load(self.config_file)
        
        # 5. 비교
        assert loaded_config.sr == config.sr
        assert loaded_config.n_mfcc == config.n_mfcc
        assert loaded_config.data_directory == config.data_directory
        
        # 6. 업데이트
        loaded_config.update({"sr": 48000})
        assert loaded_config.sr == 48000
        
        # 7. 재검증
        loaded_config.validate()


if __name__ == "__main__":
    pytest.main([__file__])