"""
pytest 설정 및 공통 픽스처

전체 테스트 스위트에서 사용할 공통 설정과 픽스처들을 정의합니다.
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, DEFAULT_CONFIG


@pytest.fixture(scope="session")
def project_root():
    """프로젝트 루트 디렉토리 경로"""
    return os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def temp_data_dir():
    """세션 레벨 임시 데이터 디렉토리"""
    temp_dir = tempfile.mkdtemp(prefix="wm_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """테스트별 임시 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config(temp_dir):
    """테스트용 설정 객체"""
    return Config(
        sr=22050,
        n_mfcc=13,
        n_chroma=12,
        data_directory=temp_dir,
        noise_directory=temp_dir,
        snr_levels=[0, 5, 10],
        classes=["watermelon_A", "watermelon_B", "watermelon_C"]
    )


@pytest.fixture
def sample_audio_data():
    """테스트용 샘플 오디오 데이터"""
    sr = 22050
    duration = 2.0
    n_samples = int(sr * duration)
    
    # 440Hz 사인파
    t = np.linspace(0, duration, n_samples)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    return {
        'signal': signal,
        'sr': sr,
        'duration': duration,
        'n_samples': n_samples
    }


@pytest.fixture
def sample_audio_file(temp_dir, sample_audio_data):
    """테스트용 샘플 오디오 파일"""
    filepath = os.path.join(temp_dir, "sample.wav")
    sf.write(filepath, sample_audio_data['signal'], sample_audio_data['sr'])
    return filepath


@pytest.fixture
def sample_noise_data():
    """테스트용 샘플 노이즈 데이터"""
    sr = 22050
    duration = 2.0
    n_samples = int(sr * duration)
    
    # 가우시안 백색 노이즈
    noise = np.random.normal(0, 0.1, n_samples)
    
    return {
        'noise': noise,
        'sr': sr,
        'duration': duration,
        'n_samples': n_samples
    }


@pytest.fixture
def sample_noise_file(temp_dir, sample_noise_data):
    """테스트용 샘플 노이즈 파일"""
    filepath = os.path.join(temp_dir, "noise.wav")
    sf.write(filepath, sample_noise_data['noise'], sample_noise_data['sr'])
    return filepath


@pytest.fixture
def test_dataset_structure(temp_dir):
    """테스트용 데이터셋 구조 생성"""
    # 디렉토리 구조 생성
    data_dir = os.path.join(temp_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    noise_dir = os.path.join(data_dir, "noise")
    
    # 클래스별 디렉토리
    classes = ["watermelon_A", "watermelon_B", "watermelon_C"]
    splits = ["train", "validation", "test"]
    
    structure = {
        'root': data_dir,
        'raw': raw_dir,
        'noise': noise_dir,
        'classes': {},
        'files': []
    }
    
    for split in splits:
        for class_name in classes:
            class_dir = os.path.join(raw_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            if class_name not in structure['classes']:
                structure['classes'][class_name] = {}
            structure['classes'][class_name][split] = class_dir
            
            # 각 클래스/분할에 샘플 파일 생성
            num_files = 5 if split == "train" else 2
            for i in range(num_files):
                # 클래스별로 다른 주파수 사용
                freq_offset = hash(class_name) % 200
                freq = 440 + freq_offset
                
                t = np.linspace(0, 1.5, int(22050 * 1.5))
                signal = 0.4 * np.sin(2 * np.pi * freq * t)
                
                filename = f"{class_name}_{split}_{i:02d}.wav"
                filepath = os.path.join(class_dir, filename)
                sf.write(filepath, signal, 22050)
                
                structure['files'].append(filepath)
    
    # 노이즈 파일 생성
    os.makedirs(noise_dir, exist_ok=True)
    for i in range(3):
        noise_data = np.random.normal(0, 0.08, int(22050 * 1.5))
        filename = f"noise_{i}.wav"
        filepath = os.path.join(noise_dir, filename)
        sf.write(filepath, noise_data, 22050)
        structure['files'].append(filepath)
    
    return structure


@pytest.fixture
def feature_vectors():
    """테스트용 특징 벡터 데이터"""
    np.random.seed(42)  # 재현 가능한 결과
    
    n_samples = 100
    n_features = 30  # MFCC(13) + 통계(5) + Chroma(12)
    
    # 클래스별로 약간 다른 분포를 가진 특징 생성
    features = []
    labels = []
    
    for class_idx, class_name in enumerate(["watermelon_A", "watermelon_B", "watermelon_C"]):
        # 클래스별 중심점 설정
        center = np.random.randn(n_features) + class_idx * 2
        
        for _ in range(n_samples // 3):
            # 중심점 주변에 노이즈 추가
            feature = center + np.random.normal(0, 0.5, n_features)
            features.append(feature)
            labels.append(class_idx)
    
    return {
        'features': np.array(features),
        'labels': np.array(labels),
        'class_names': ["watermelon_A", "watermelon_B", "watermelon_C"],
        'n_features': n_features
    }


@pytest.fixture
def mock_trained_models():
    """테스트용 mock 훈련된 모델들"""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # 간단한 데이터셋 생성
    np.random.seed(42)
    X = np.random.randn(150, 30)
    y = np.random.randint(0, 3, 150)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    # 모델 훈련
    svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    return {
        'svm': svm_model,
        'random_forest': rf_model,
        'test_data': (X_test, y_test),
        'train_data': (X_train, y_train)
    }


# 테스트 마커 정의
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# 테스트 수집 시 실행
def pytest_collection_modifyitems(config, items):
    """테스트 항목 수정"""
    for item in items:
        # 성능 테스트는 slow 마커 추가
        if "performance" in str(item.fspath) or "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
        
        # 통합 테스트는 integration 마커 추가
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # 단위 테스트는 unit 마커 추가
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# 에러 처리를 위한 픽스처
@pytest.fixture(autouse=True)
def setup_test_environment():
    """모든 테스트에 자동으로 적용되는 환경 설정"""
    # NumPy 에러 설정
    original_settings = np.seterr(all='raise')
    
    yield
    
    # 원래 설정 복원
    np.seterr(**original_settings)


# 로깅 설정
@pytest.fixture(autouse=True)
def setup_logging():
    """테스트용 로깅 설정"""
    import logging
    
    # 테스트 중에는 로그 레벨을 WARNING 이상으로 설정
    logging.getLogger().setLevel(logging.WARNING)
    
    # 특정 라이브러리의 verbose 로그 억제
    logging.getLogger('librosa').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    
    yield
    
    # 로그 레벨 복원
    logging.getLogger().setLevel(logging.INFO)


# 테스트 실행 전후 훅
def pytest_runtest_setup(item):
    """각 테스트 실행 전 호출"""
    # 메모리 사용량이 많은 테스트의 경우 가비지 컬렉션 수행
    if hasattr(item, 'get_closest_marker'):
        if item.get_closest_marker('performance') or item.get_closest_marker('slow'):
            import gc
            gc.collect()


def pytest_runtest_teardown(item, nextitem):
    """각 테스트 실행 후 호출"""
    # 성능 테스트 후 메모리 정리
    if hasattr(item, 'get_closest_marker'):
        if item.get_closest_marker('performance') or item.get_closest_marker('slow'):
            import gc
            gc.collect()


# 실패한 테스트에 대한 정보 수집
def pytest_runtest_makereport(item, call):
    """테스트 실행 결과 리포트 생성"""
    if call.when == "call" and call.excinfo is not None:
        # 실패한 테스트의 추가 정보 수집
        test_name = item.nodeid
        error_info = str(call.excinfo.value)
        
        # 로그 파일에 기록 (실제 구현에서는 적절한 로깅 시스템 사용)
        print(f"Test failed: {test_name}")
        print(f"Error: {error_info}")


# 병렬 테스트 지원을 위한 픽스처
@pytest.fixture(scope="session")
def worker_id(request):
    """pytest-xdist 워커 ID (병렬 실행시)"""
    if hasattr(request.config, 'workerinput'):
        return request.config.workerinput['workerid']
    return 'master'