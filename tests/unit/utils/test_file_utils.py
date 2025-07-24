"""
파일 유틸리티 모듈에 대한 단위 테스트

이 모듈은 src/utils/file_utils.py의 모든 유틸리티 클래스를 테스트합니다.
"""

import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.file_utils import (
    FileUtils, JsonUtils, PickleUtils, AudioFileUtils,
    ArrayUtils, VisualizationUtils, MemoryUtils
)


class TestFileUtils:
    """FileUtils 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("테스트 파일 내용")
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ensure_directory_exists(self):
        """디렉토리 생성 테스트"""
        new_dir = os.path.join(self.temp_dir, "new", "nested", "directory")
        assert not os.path.exists(new_dir)
        
        FileUtils.ensure_directory_exists(new_dir)
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
    
    def test_safe_file_read_success(self):
        """안전한 파일 읽기 성공 테스트"""
        content = FileUtils.safe_file_read(self.test_file)
        assert content == "테스트 파일 내용"
    
    def test_safe_file_read_nonexistent(self):
        """존재하지 않는 파일 읽기 테스트"""
        content = FileUtils.safe_file_read("nonexistent.txt")
        assert content is None
    
    def test_safe_file_write(self):
        """안전한 파일 쓰기 테스트"""
        new_file = os.path.join(self.temp_dir, "write_test.txt")
        content = "새로운 내용"
        
        success = FileUtils.safe_file_write(new_file, content)
        assert success is True
        assert os.path.exists(new_file)
        
        with open(new_file, 'r', encoding='utf-8') as f:
            assert f.read() == content
    
    def test_get_file_size(self):
        """파일 크기 조회 테스트"""
        size = FileUtils.get_file_size(self.test_file)
        expected_size = len("테스트 파일 내용".encode('utf-8'))
        assert size == expected_size
    
    def test_get_file_size_nonexistent(self):
        """존재하지 않는 파일 크기 조회 테스트"""
        size = FileUtils.get_file_size("nonexistent.txt")
        assert size == 0
    
    def test_list_files_with_extension(self):
        """확장자별 파일 목록 조회 테스트"""
        # 테스트 파일들 생성
        test_files = ["test1.wav", "test2.wav", "test3.mp3", "test4.txt"]
        for filename in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test")
        
        wav_files = FileUtils.list_files_with_extension(self.temp_dir, ".wav")
        assert len(wav_files) == 2
        assert all(f.endswith('.wav') for f in wav_files)
    
    def test_clean_filename(self):
        """파일명 정리 테스트"""
        dirty_name = "파일명/with\\invalid:chars|<>?*.txt"
        clean_name = FileUtils.clean_filename(dirty_name)
        
        # 유효하지 않은 문자들이 제거되었는지 확인
        invalid_chars = ['/', '\\', ':', '|', '<', '>', '?', '*']
        for char in invalid_chars:
            assert char not in clean_name
    
    def test_backup_file(self):
        """파일 백업 테스트"""
        backup_path = FileUtils.backup_file(self.test_file)
        
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.bak')
        
        # 백업 파일 내용 확인
        with open(backup_path, 'r', encoding='utf-8') as f:
            assert f.read() == "테스트 파일 내용"


class TestJsonUtils:
    """JsonUtils 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            "name": "수박 분류기",
            "version": "1.0",
            "config": {
                "sr": 22050,
                "hop_length": 512,
                "classes": ["watermelon_A", "watermelon_B", "watermelon_C"]
            }
        }
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_json(self):
        """JSON 저장 및 로딩 테스트"""
        json_file = os.path.join(self.temp_dir, "test.json")
        
        # 저장 테스트
        success = JsonUtils.save_json(self.test_data, json_file)
        assert success is True
        assert os.path.exists(json_file)
        
        # 로딩 테스트
        loaded_data = JsonUtils.load_json(json_file)
        assert loaded_data == self.test_data
    
    def test_load_nonexistent_json(self):
        """존재하지 않는 JSON 파일 로딩 테스트"""
        result = JsonUtils.load_json("nonexistent.json")
        assert result is None
    
    def test_validate_json_schema(self):
        """JSON 스키마 검증 테스트"""
        schema = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "config": {"type": "object"}
            }
        }
        
        # 유효한 데이터 테스트
        assert JsonUtils.validate_json_schema(self.test_data, schema) is True
        
        # 무효한 데이터 테스트 (필수 필드 누락)
        invalid_data = {"name": "test"}
        assert JsonUtils.validate_json_schema(invalid_data, schema) is False
    
    def test_pretty_print_json(self):
        """JSON 예쁘게 출력 테스트"""
        output = JsonUtils.pretty_print_json(self.test_data)
        
        # 출력에 들여쓰기와 줄바꿈이 포함되어야 함
        assert '\n' in output
        assert '  ' in output  # 들여쓰기
        assert '"name": "수박 분류기"' in output


class TestPickleUtils:
    """PickleUtils 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            "array": np.array([1, 2, 3, 4, 5]),
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_pickle(self):
        """Pickle 저장 및 로딩 테스트"""
        pickle_file = os.path.join(self.temp_dir, "test.pkl")
        
        # 저장 테스트
        success = PickleUtils.save_pickle(self.test_data, pickle_file)
        assert success is True
        assert os.path.exists(pickle_file)
        
        # 로딩 테스트
        loaded_data = PickleUtils.load_pickle(pickle_file)
        
        # NumPy 배열은 별도로 비교
        np.testing.assert_array_equal(loaded_data["array"], self.test_data["array"])
        assert loaded_data["list"] == self.test_data["list"]
        assert loaded_data["dict"] == self.test_data["dict"]
    
    def test_load_nonexistent_pickle(self):
        """존재하지 않는 Pickle 파일 로딩 테스트"""
        result = PickleUtils.load_pickle("nonexistent.pkl")
        assert result is None
    
    def test_get_pickle_info(self):
        """Pickle 파일 정보 조회 테스트"""
        pickle_file = os.path.join(self.temp_dir, "test.pkl")
        PickleUtils.save_pickle(self.test_data, pickle_file)
        
        info = PickleUtils.get_pickle_info(pickle_file)
        
        assert "file_size" in info
        assert "created_time" in info
        assert "data_type" in info
        assert info["data_type"] == "dict"


class TestAudioFileUtils:
    """AudioFileUtils 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 실행되는 설정"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """각 테스트 후에 실행되는 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_valid_audio_extension(self):
        """유효한 오디오 확장자 확인 테스트"""
        valid_extensions = [".wav", ".mp3", ".flac", ".aac", ".ogg"]
        invalid_extensions = [".txt", ".jpg", ".pdf", ".docx"]
        
        for ext in valid_extensions:
            assert AudioFileUtils.is_valid_audio_extension(ext) is True
        
        for ext in invalid_extensions:
            assert AudioFileUtils.is_valid_audio_extension(ext) is False
    
    def test_get_audio_files_recursive(self):
        """재귀적 오디오 파일 검색 테스트"""
        # 테스트 디렉토리 구조 생성
        subdir1 = os.path.join(self.temp_dir, "subdir1")
        subdir2 = os.path.join(self.temp_dir, "subdir2")
        os.makedirs(subdir1)
        os.makedirs(subdir2)
        
        # 오디오 파일들 생성
        audio_files = [
            os.path.join(self.temp_dir, "test1.wav"),
            os.path.join(subdir1, "test2.mp3"),
            os.path.join(subdir2, "test3.flac"),
            os.path.join(self.temp_dir, "test.txt")  # 비오디오 파일
        ]
        
        for filepath in audio_files:
            with open(filepath, 'w') as f:
                f.write("test")
        
        found_files = AudioFileUtils.get_audio_files_recursive(self.temp_dir)
        
        # 3개의 오디오 파일만 찾아져야 함
        assert len(found_files) == 3
        assert all(AudioFileUtils.is_valid_audio_extension(
            os.path.splitext(f)[1]) for f in found_files)
    
    @patch('librosa.get_duration')
    def test_get_audio_duration(self, mock_duration):
        """오디오 지속시간 조회 테스트"""
        mock_duration.return_value = 2.5
        
        test_file = os.path.join(self.temp_dir, "test.wav")
        with open(test_file, 'w') as f:
            f.write("fake audio data")
        
        duration = AudioFileUtils.get_audio_duration(test_file)
        assert duration == 2.5
        mock_duration.assert_called_once_with(filename=test_file)
    
    def test_generate_audio_filename(self):
        """오디오 파일명 생성 테스트"""
        base_name = "수박A_레오-001"
        noise_type = "homeplus"
        snr_level = 5.0
        
        filename = AudioFileUtils.generate_audio_filename(
            base_name, noise_type, snr_level)
        
        expected = "수박A_레오-001_noise_homeplus_snr+5.0dB.wav"
        assert filename == expected
    
    def test_parse_augmented_filename(self):
        """증강된 파일명 파싱 테스트"""
        filename = "수박A_레오-001_noise_homeplus_snr+5.0dB.wav"
        
        parsed = AudioFileUtils.parse_augmented_filename(filename)
        
        assert parsed["original_name"] == "수박A_레오-001"
        assert parsed["noise_type"] == "homeplus"
        assert parsed["snr_level"] == 5.0
        assert parsed["is_augmented"] is True
    
    def test_parse_original_filename(self):
        """원본 파일명 파싱 테스트"""
        filename = "수박A_레오-001.wav"
        
        parsed = AudioFileUtils.parse_augmented_filename(filename)
        
        assert parsed["original_name"] == "수박A_레오-001"
        assert parsed["noise_type"] is None
        assert parsed["snr_level"] is None
        assert parsed["is_augmented"] is False


class TestArrayUtils:
    """ArrayUtils 클래스 테스트"""
    
    def test_normalize_array(self):
        """배열 정규화 테스트"""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        
        # Min-Max 정규화
        normalized = ArrayUtils.normalize_array(arr, method='minmax')
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        
        # Z-score 정규화
        z_normalized = ArrayUtils.normalize_array(arr, method='zscore')
        assert abs(z_normalized.mean()) < 1e-10  # 평균이 0에 가까워야 함
        assert abs(z_normalized.std() - 1.0) < 1e-10  # 표준편차가 1에 가까워야 함
    
    def test_split_array(self):
        """배열 분할 테스트"""
        arr = np.arange(100)
        
        train, val, test = ArrayUtils.split_array(
            arr, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        
        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10
        
        # 모든 원소가 포함되어야 함
        combined = np.concatenate([train, val, test])
        assert len(np.unique(combined)) == 100
    
    def test_batch_process(self):
        """배치 처리 테스트"""
        arr = np.arange(100)
        batch_size = 10
        
        batches = list(ArrayUtils.batch_process(arr, batch_size))
        
        assert len(batches) == 10
        assert all(len(batch) == batch_size for batch in batches)
    
    def test_remove_outliers(self):
        """이상치 제거 테스트"""
        # 정상 데이터와 이상치가 포함된 배열
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([10, -10, 15])  # 명백한 이상치
        arr = np.concatenate([normal_data, outliers])
        
        cleaned = ArrayUtils.remove_outliers(arr, method='iqr')
        
        # 이상치가 제거되어 배열 크기가 작아져야 함
        assert len(cleaned) < len(arr)
        # 큰 이상치들이 제거되어야 함
        assert not np.any(cleaned > 5)
        assert not np.any(cleaned < -5)
    
    def test_calculate_feature_importance(self):
        """특징 중요도 계산 테스트"""
        # 간단한 선형 관계 데이터
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # 첫 번째와 세 번째 특징이 중요하도록 설정
        y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.randn(100) * 0.1
        
        importance = ArrayUtils.calculate_feature_importance(X, y)
        
        assert len(importance) == 5
        # 첫 번째와 세 번째 특징의 중요도가 높아야 함
        assert importance[0] > importance[1]
        assert importance[2] > importance[1]


class TestMemoryUtils:
    """MemoryUtils 클래스 테스트"""
    
    def test_get_memory_usage(self):
        """메모리 사용량 조회 테스트"""
        usage = MemoryUtils.get_memory_usage()
        
        assert "total" in usage
        assert "available" in usage
        assert "percent" in usage
        assert "used" in usage
        
        # 모든 값이 양수여야 함
        assert all(v >= 0 for v in usage.values())
        assert usage["percent"] <= 100.0
    
    def test_chunk_generator(self):
        """청크 생성기 테스트"""
        data = list(range(100))
        chunk_size = 10
        
        chunks = list(MemoryUtils.chunk_generator(data, chunk_size))
        
        assert len(chunks) == 10
        assert all(len(chunk) == chunk_size for chunk in chunks)
        
        # 모든 데이터가 포함되어야 함
        flattened = [item for chunk in chunks for item in chunk]
        assert flattened == data
    
    def test_memory_efficient_operation(self):
        """메모리 효율적 연산 테스트"""
        def simple_operation(x):
            return x * 2
        
        data = list(range(100))
        chunk_size = 10
        
        results = list(MemoryUtils.memory_efficient_operation(
            data, simple_operation, chunk_size))
        
        expected = [x * 2 for x in data]
        assert results == expected
    
    @patch('gc.collect')
    def test_force_garbage_collection(self, mock_gc):
        """가비지 컬렉션 강제 실행 테스트"""
        MemoryUtils.force_garbage_collection()
        mock_gc.assert_called_once()


# 테스트 실행을 위한 픽스처
@pytest.fixture
def sample_audio_data():
    """테스트용 샘플 오디오 데이터"""
    return {
        "sr": 22050,
        "duration": 2.0,
        "n_samples": 44100,
        "features": np.random.randn(30)
    }


@pytest.fixture
def temp_directory():
    """임시 디렉토리 픽스처"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])