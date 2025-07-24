# API 참조 문서

## 개요

이 문서는 수박 소리 분류 시스템의 모든 공개 API에 대한 상세한 참조를 제공합니다.

## 목차

1. [Audio 모듈](#audio-모듈)
   - [AudioFeatureExtractor](#audiofeatureextractor)
   - [FeatureVector](#featurevector)
2. [Data 모듈](#data-모듈)
   - [DataPipeline](#datapipeline)
   - [AudioAugmentor](#audioaugmentor)
   - [BatchAugmentor](#batchaugmentor)
3. [ML 모듈](#ml-모듈)
   - [ModelTrainer](#modeltrainer)
   - [ModelEvaluator](#modelevaluator)
   - [ModelConverter](#modelconverter)
4. [Utils 모듈](#utils-모듈)
   - [FileUtils](#fileutils)
   - [ArrayUtils](#arrayutils)
   - [VisualizationUtils](#visualizationutils)
   - [PerformanceMonitor](#performancemonitor)

---

## Audio 모듈

### AudioFeatureExtractor

오디오 파일에서 머신러닝에 사용할 특징을 추출하는 클래스입니다.

#### 클래스 정의

```python
class AudioFeatureExtractor(LoggerMixin)
```

#### 생성자

```python
def __init__(self, config: Config = None)
```

**매개변수:**
- `config` (Config, 선택): 특징 추출 설정. 기본값은 DEFAULT_CONFIG

#### 주요 메서드

##### extract_features

```python
def extract_features(self, audio_path: str) -> np.ndarray
```

오디오 파일에서 30차원 특징 벡터를 추출합니다.

**매개변수:**
- `audio_path` (str): 오디오 파일 경로 (.wav 형식)

**반환값:**
- `np.ndarray`: 30차원 특징 벡터 [MFCC(13) + Mel통계(2) + Spectral(3) + Chroma(12)]

**예외:**
- `FileNotFoundError`: 오디오 파일을 찾을 수 없음
- `ValueError`: 오디오 파일이 손상되었거나 형식이 잘못됨
- `RuntimeError`: librosa 로딩 실패

**사용 예제:**
```python
extractor = AudioFeatureExtractor()
features = extractor.extract_features("audio.wav")
print(f"특징 벡터 크기: {features.shape}")  # (30,)
```

##### extract_features_detailed

```python
def extract_features_detailed(self, audio_path: str) -> FeatureVector
```

오디오 파일에서 구조화된 특징 객체를 추출합니다.

**매개변수:**
- `audio_path` (str): 오디오 파일 경로

**반환값:**
- `FeatureVector`: 구조화된 특징 객체

**사용 예제:**
```python
feature_obj = extractor.extract_features_detailed("audio.wav")
print(f"MFCC: {feature_obj.mfcc}")
print(f"Spectral Centroid: {feature_obj.spectral_centroid}")
```

##### validate_audio_file

```python
def validate_audio_file(self, audio_path: str) -> dict
```

오디오 파일의 유효성을 검사합니다.

**매개변수:**
- `audio_path` (str): 검사할 오디오 파일 경로

**반환값:**
- `dict`: 검증 결과
  - `is_valid` (bool): 파일이 유효한지 여부
  - `duration` (float): 오디오 길이 (초)
  - `sample_rate` (int): 샘플링 레이트
  - `channels` (int): 채널 수
  - `warnings` (list): 경고 메시지들
  - `errors` (list): 오류 메시지들

---

### FeatureVector

오디오 특징을 구조화된 형태로 저장하는 데이터 클래스입니다.

#### 클래스 정의

```python
@dataclass
class FeatureVector
```

#### 속성

- `mfcc` (np.ndarray): 13차원 MFCC 특징
- `mel_mean` (float): Mel spectrogram 평균
- `mel_std` (float): Mel spectrogram 표준편차
- `spectral_centroid` (float): 스펙트럼 중심점
- `spectral_rolloff` (float): 스펙트럼 롤오프
- `zero_crossing_rate` (float): 영점 교차율
- `chroma` (np.ndarray): 12차원 Chroma 특징

#### 주요 메서드

##### to_array

```python
def to_array(self) -> np.ndarray
```

모든 특징을 하나의 1차원 배열로 연결합니다.

**반환값:**
- `np.ndarray`: 30차원 특징 벡터

##### feature_names

```python
@property
def feature_names(self) -> list
```

각 특징의 이름을 반환합니다.

**반환값:**
- `list`: 30개 특징 이름들

---

## Data 모듈

### DataPipeline

데이터 로딩, 분할, 증강을 관리하는 파이프라인 클래스입니다.

#### 클래스 정의

```python
class DataPipeline(LoggerMixin)
```

#### 생성자

```python
def __init__(self, config: Config = None)
```

#### 주요 메서드

##### run_complete_pipeline

```python
def run_complete_pipeline(self) -> dict
```

전체 데이터 파이프라인을 실행합니다.

**반환값:**
- `dict`: 파이프라인 실행 결과
  - `train_data` (dict): 훈련 데이터와 특징
  - `validation_data` (dict): 검증 데이터와 특징
  - `test_data` (dict): 테스트 데이터와 특징
  - `metadata` (dict): 파이프라인 메타데이터

##### load_train_data

```python
def load_train_data(self) -> dict
```

훈련 데이터를 로드합니다.

**반환값:**
- `dict`: 클래스별 오디오 파일 경로들

---

### AudioAugmentor

단일 오디오 파일에 대한 소음 증강을 수행하는 클래스입니다.

#### 클래스 정의

```python
class AudioAugmentor(LoggerMixin)
```

#### 주요 메서드

##### augment_noise

```python
def augment_noise(self, clean_audio_path: str, noise_audio_path: str, 
                  output_path: str, snr_db: float) -> bool
```

깨끗한 오디오에 지정된 SNR로 소음을 추가합니다.

**매개변수:**
- `clean_audio_path` (str): 원본 오디오 파일 경로
- `noise_audio_path` (str): 소음 파일 경로
- `output_path` (str): 출력 파일 경로
- `snr_db` (float): 목표 SNR (dB)

**반환값:**
- `bool`: 증강 성공 여부

##### calculate_snr

```python
def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float
```

신호와 소음 간의 SNR을 계산합니다.

**매개변수:**
- `signal` (np.ndarray): 신호 배열
- `noise` (np.ndarray): 소음 배열

**반환값:**
- `float`: SNR 값 (dB)

---

### BatchAugmentor

여러 오디오 파일에 대한 배치 증강을 수행하는 클래스입니다.

#### 클래스 정의

```python
class BatchAugmentor(LoggerMixin)
```

#### 주요 메서드

##### augment_class_directory

```python
def augment_class_directory(self, input_dir: str, output_dir: str, 
                           target_multiplier: int = None) -> list
```

디렉토리 내 모든 오디오 파일을 증강합니다.

**매개변수:**
- `input_dir` (str): 입력 디렉토리 경로
- `output_dir` (str): 출력 디렉토리 경로
- `target_multiplier` (int, 선택): 증강 배수

**반환값:**
- `list`: 생성된 증강 파일 정보들

---

## ML 모듈

### ModelTrainer

머신러닝 모델 훈련을 담당하는 클래스입니다.

#### 클래스 정의

```python
class ModelTrainer(LoggerMixin)
```

#### 주요 메서드

##### train_with_cv

```python
def train_with_cv(self, X_train: np.ndarray, y_train: np.ndarray, 
                  cv_folds: int = 5) -> dict
```

교차 검증을 사용하여 모델을 훈련합니다.

**매개변수:**
- `X_train` (np.ndarray): 훈련 특징 행렬
- `y_train` (np.ndarray): 훈련 레이블
- `cv_folds` (int): 교차 검증 폴드 수

**반환값:**
- `dict`: 훈련 결과
  - 각 모델별 TrainingResult 객체

##### save_models

```python
def save_models(self, output_dir: str) -> dict
```

훈련된 모델들을 저장합니다.

**매개변수:**
- `output_dir` (str): 모델 저장 디렉토리

**반환값:**
- `dict`: 저장된 모델 경로들

##### load_model

```python
def load_model(self, model_path: str) -> object
```

저장된 모델을 로드합니다.

**매개변수:**
- `model_path` (str): 모델 파일 경로

**반환값:**
- `object`: 로드된 scikit-learn 모델

---

### ModelEvaluator

모델 성능 평가를 담당하는 클래스입니다.

#### 클래스 정의

```python
class ModelEvaluator(LoggerMixin)
```

#### 주요 메서드

##### evaluate_model

```python
def evaluate_model(self, model: object, X_test: np.ndarray, 
                   y_test: np.ndarray, model_name: str) -> ClassificationMetrics
```

단일 모델의 성능을 평가합니다.

**매개변수:**
- `model` (object): 평가할 모델
- `X_test` (np.ndarray): 테스트 특징
- `y_test` (np.ndarray): 테스트 레이블
- `model_name` (str): 모델 이름

**반환값:**
- `ClassificationMetrics`: 평가 메트릭

##### compare_models

```python
def compare_models(self, models: dict, X_test: np.ndarray, 
                   y_test: np.ndarray) -> ModelComparison
```

여러 모델의 성능을 비교합니다.

**매개변수:**
- `models` (dict): 모델명과 모델 객체 매핑
- `X_test` (np.ndarray): 테스트 특징
- `y_test` (np.ndarray): 테스트 레이블

**반환값:**
- `ModelComparison`: 모델 비교 결과

---

### ModelConverter

모델을 다른 형식으로 변환하는 클래스입니다.

#### 클래스 정의

```python
class ModelConverter(LoggerMixin)
```

#### 주요 메서드

##### convert_to_coreml

```python
def convert_to_coreml(self, model_path: str, output_path: str, 
                      model_name: str = None, 
                      model_description: str = None) -> ConversionResult
```

Pickle 모델을 Core ML 형식으로 변환합니다.

**매개변수:**
- `model_path` (str): 입력 Pickle 모델 경로
- `output_path` (str): 출력 Core ML 모델 경로
- `model_name` (str, 선택): 모델 이름
- `model_description` (str, 선택): 모델 설명

**반환값:**
- `ConversionResult`: 변환 결과

##### validate_model_conversion

```python
def validate_model_conversion(self, pickle_path: str, coreml_path: str, 
                            test_samples: np.ndarray = None) -> bool
```

변환된 모델의 정확성을 검증합니다.

**매개변수:**
- `pickle_path` (str): 원본 Pickle 모델 경로
- `coreml_path` (str): 변환된 Core ML 모델 경로
- `test_samples` (np.ndarray, 선택): 테스트 샘플

**반환값:**
- `bool`: 검증 통과 여부

---

## Utils 모듈

### FileUtils

파일 및 디렉토리 작업을 위한 유틸리티 클래스입니다.

#### 정적 메서드

##### ensure_directory

```python
@staticmethod
def ensure_directory(directory_path: str) -> str
```

디렉토리가 존재하지 않으면 생성합니다.

**매개변수:**
- `directory_path` (str): 생성할 디렉토리 경로

**반환값:**
- `str`: 생성된 디렉토리 경로

##### find_files

```python
@staticmethod
def find_files(directory: str, pattern: str = "*", 
               recursive: bool = True, file_types: list = None) -> list
```

지정된 패턴과 일치하는 파일들을 찾습니다.

**매개변수:**
- `directory` (str): 검색할 디렉토리
- `pattern` (str): 파일 패턴
- `recursive` (bool): 재귀 검색 여부
- `file_types` (list, 선택): 허용할 파일 확장자

**반환값:**
- `list`: 찾은 파일 경로들

##### get_file_info

```python
@staticmethod
def get_file_info(file_path: str) -> dict
```

파일의 상세 정보를 반환합니다.

**매개변수:**
- `file_path` (str): 파일 경로

**반환값:**
- `dict`: 파일 정보
  - `size_bytes` (int): 파일 크기 (바이트)
  - `size_mb` (float): 파일 크기 (MB)
  - `created_time` (str): 생성 시간
  - `modified_time` (str): 수정 시간
  - `extension` (str): 파일 확장자

---

### ArrayUtils

배열 처리를 위한 유틸리티 클래스입니다.

#### 정적 메서드

##### normalize_features

```python
@staticmethod
def normalize_features(X: np.ndarray, method: str = 'standard') -> tuple
```

특징을 정규화합니다.

**매개변수:**
- `X` (np.ndarray): 입력 특징 행렬
- `method` (str): 정규화 방법 ('standard', 'minmax', 'robust')

**반환값:**
- `tuple`: (정규화된 데이터, 정규화 파라미터)

##### split_array

```python
@staticmethod
def split_array(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                stratify: bool = True) -> tuple
```

데이터를 훈련/테스트 세트로 분할합니다.

**매개변수:**
- `X` (np.ndarray): 특징 행렬
- `y` (np.ndarray): 레이블 배열
- `test_size` (float): 테스트 세트 비율
- `stratify` (bool): 계층화 분할 여부

**반환값:**
- `tuple`: (X_train, X_test, y_train, y_test)

##### compute_feature_importance

```python
@staticmethod
def compute_feature_importance(X: np.ndarray, y: np.ndarray, 
                              feature_names: list = None) -> dict
```

특징 중요도를 계산합니다.

**매개변수:**
- `X` (np.ndarray): 특징 행렬
- `y` (np.ndarray): 레이블 배열
- `feature_names` (list, 선택): 특징 이름들

**반환값:**
- `dict`: 특징별 중요도 점수

---

### VisualizationUtils

데이터 시각화를 위한 유틸리티 클래스입니다.

#### 정적 메서드

##### plot_feature_distribution

```python
@staticmethod
def plot_feature_distribution(X: np.ndarray, feature_names: list = None, 
                             output_path: str = None) -> None
```

특징들의 분포를 시각화합니다.

**매개변수:**
- `X` (np.ndarray): 특징 행렬
- `feature_names` (list, 선택): 특징 이름들
- `output_path` (str, 선택): 저장할 파일 경로

##### plot_correlation_matrix

```python
@staticmethod
def plot_correlation_matrix(X: np.ndarray, feature_names: list = None,
                           output_path: str = None) -> None
```

특징 간 상관관계 매트릭스를 시각화합니다.

**매개변수:**
- `X` (np.ndarray): 특징 행렬
- `feature_names` (list, 선택): 특징 이름들
- `output_path` (str, 선택): 저장할 파일 경로

##### plot_class_distribution

```python
@staticmethod
def plot_class_distribution(y: np.ndarray, class_names: list = None,
                           output_path: str = None) -> None
```

클래스 분포를 시각화합니다.

**매개변수:**
- `y` (np.ndarray): 레이블 배열
- `class_names` (list, 선택): 클래스 이름들
- `output_path` (str, 선택): 저장할 파일 경로

---

### PerformanceMonitor

시스템 성능을 모니터링하는 클래스입니다.

#### 클래스 정의

```python
class PerformanceMonitor(LoggerMixin)
```

#### 주요 메서드

##### start_step_monitoring

```python
def start_step_monitoring(self, step_name: str) -> None
```

특정 단계의 성능 모니터링을 시작합니다.

**매개변수:**
- `step_name` (str): 모니터링할 단계 이름

##### end_step_monitoring

```python
def end_step_monitoring(self, step_name: str, additional_info: dict = None) -> dict
```

단계 모니터링을 종료하고 결과를 반환합니다.

**매개변수:**
- `step_name` (str): 종료할 단계 이름
- `additional_info` (dict, 선택): 추가 정보

**반환값:**
- `dict`: 성능 메트릭

##### benchmark_operation

```python
def benchmark_operation(self, func: callable, operation_name: str, 
                       *args, **kwargs) -> tuple
```

특정 연산의 성능을 벤치마킹합니다.

**매개변수:**
- `func` (callable): 벤치마킹할 함수
- `operation_name` (str): 연산 이름
- `*args, **kwargs`: 함수 인자들

**반환값:**
- `tuple`: (함수 반환값, 성능 메트릭)

##### get_performance_summary

```python
def get_performance_summary(self) -> dict
```

전체 성능 요약을 반환합니다.

**반환값:**
- `dict`: 성능 요약 정보

---

## 데이터 클래스

### TrainingResult

모델 훈련 결과를 저장하는 데이터 클래스입니다.

```python
@dataclass
class TrainingResult:
    model_name: str
    best_score: float
    best_params: dict
    cv_scores: list
    training_time: float
    model: object
```

### ClassificationMetrics

분류 모델의 성능 메트릭을 저장하는 데이터 클래스입니다.

```python
@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: dict
    recall: dict
    f1_score: dict
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    roc_auc: float
    pr_auc: float
```

### ConversionResult

모델 변환 결과를 저장하는 데이터 클래스입니다.

```python
@dataclass
class ConversionResult:
    success: bool
    output_path: str
    model_size_mb: float
    conversion_time: float
    validation_passed: bool
    validation_score: float
    error_message: str
```

---

## 예외 클래스

시스템에서 사용하는 사용자 정의 예외들입니다.

### AudioProcessingError

```python
class AudioProcessingError(Exception):
    """오디오 처리 중 발생하는 오류"""
    pass
```

### ModelTrainingError

```python
class ModelTrainingError(Exception):
    """모델 훈련 중 발생하는 오류"""
    pass
```

### FeatureExtractionError

```python
class FeatureExtractionError(Exception):
    """특징 추출 중 발생하는 오류"""
    pass
```

---

## 사용 패턴

### 기본 사용 패턴

```python
from src.audio.feature_extraction import AudioFeatureExtractor
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator

# 1. 특징 추출
extractor = AudioFeatureExtractor()
features = extractor.extract_features("audio.wav")

# 2. 모델 훈련
trainer = ModelTrainer()
results = trainer.train_with_cv(X_train, y_train)

# 3. 모델 평가
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test, "SVM")
```

### 고급 사용 패턴

```python
from src.data.pipeline import DataPipeline
from main import WatermelonClassificationPipeline

# 전체 파이프라인 실행
pipeline = WatermelonClassificationPipeline()
results = pipeline.run()
```

---

## 구성 설정

시스템 설정은 `config/config.py`의 `Config` 클래스를 통해 관리됩니다.

### Config 클래스

```python
@dataclass
class Config:
    # 데이터 경로
    data_root_dir: str = "data"
    raw_data_dir: str = "data/raw"
    noise_dir: str = "data/noise"
    
    # 오디오 처리
    sample_rate: int = 22050
    hop_length: int = 512
    n_mfcc: int = 13
    n_chroma: int = 12
    
    # 데이터 증강
    snr_levels: List[float] = field(default_factory=lambda: [-5, 0, 5, 10])
    augmentation_factor: int = 4
    
    # 모델 훈련
    cv_folds: int = 5
    random_state: int = 42
```

이 API 참조는 시스템의 모든 공개 인터페이스를 포괄하며, 개발자가 시스템을 효과적으로 사용할 수 있도록 돕습니다.