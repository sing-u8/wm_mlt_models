# 수박 소리 분류 머신러닝 시스템

이 프로젝트는 수박의 숙성도에 따른 소리를 분류하는 파이썬 기반 머신러닝 파이프라인입니다. 시스템은 librosa를 활용한 포괄적인 오디오 특징 추출, SNR 제어 소음 증강 기법, scikit-learn 기반 모델 훈련을 통해 실제 환경에서도 강건한 분류 성능을 제공합니다.

## 🎯 주요 특징

- **포괄적인 오디오 특징 추출**: MFCC, Mel Spectrogram, Spectral, Chroma 특징 (30차원)
- **SNR 제어 데이터 증강**: 실제 환경 소음을 활용한 강건한 모델 훈련
- **다중 모델 지원**: SVM, Random Forest 모델과 하이퍼파라미터 최적화
- **Cross-platform 배포**: Pickle과 Core ML 형식 지원 (iOS/macOS 호환)
- **완전한 파이프라인**: 데이터 로딩부터 모델 배포까지 자동화
- **성능 모니터링**: 실시간 메모리/CPU 사용량 및 데이터 무결성 검사
- **확장 가능한 구조**: 모듈형 설계로 새로운 모델과 특징 추가 용이

## 프로젝트 구조

프로젝트는 다음과 같은 표준화된 디렉토리 구조를 사용합니다:

```
project_root/
├── data/                       # 데이터 저장소
│   ├── raw/                   # 원본 오디오 파일
│   │   ├── watermelon_A/      # 클래스 A (예: 덜 익은 수박)
│   │   ├── watermelon_B/      # 클래스 B (예: 적당히 익은 수박)
│   │   └── watermelon_C/      # 클래스 C (예: 잘 익은 수박)
│   ├── noise/                 # 소음 파일들
│   │   ├── environmental/     # 환경 소음
│   │   ├── mechanical/        # 기계 소음
│   │   └── background/        # 배경 소음
│   ├── processed/             # 처리된 데이터 (자동 생성)
│   │   ├── augmented/         # 증강된 오디오 파일
│   │   ├── features/          # 추출된 특징 파일
│   │   └── splits/            # 데이터 분할 정보
│   └── models/                # 훈련된 모델 (자동 생성)
│       ├── artifacts/         # 모델 메타데이터
│       ├── pickle/            # Pickle 형식 모델
│       └── coreml/            # Core ML 형식 모델
├── src/                       # 소스 코드
│   └── utils/                 # 유틸리티 함수
├── config/                    # 구성 파일
├── logs/                      # 로그 파일
└── results/                   # 실험 결과 및 보고서
```

## 데이터 준비

자세한 데이터 배치 방법은 [DATA_PLACEMENT_GUIDE.md](DATA_PLACEMENT_GUIDE.md)를 참조하세요.

### 1. 수박 소리 파일 배치

`data/raw/` 디렉토리에 수박 소리 파일들을 클래스별로 배치해주세요:

- **`data/raw/watermelon_A/`**: 수박 A 유형 (예: 덜 익은 수박)
- **`data/raw/watermelon_B/`**: 수박 B 유형 (예: 적당히 익은 수박)  
- **`data/raw/watermelon_C/`**: 수박 C 유형 (예: 잘 익은 수박)

#### 파일 명명 규칙:
```
watermelon_A_001.wav
watermelon_A_002.wav
...
watermelon_B_001.wav
watermelon_B_002.wav
...
watermelon_C_001.wav
watermelon_C_002.wav
...
```

#### 최소 요구사항:
- 각 클래스당 최소 **20개 이상**의 .wav 파일
- 총 **60개 이상**의 수박 소리 파일

### 2. 소음 파일 배치

`data/noise/` 디렉토리에 소음 파일들을 유형별로 분류하여 배치해주세요:

```
data/noise/
├── environmental/           # 환경 소음
│   ├── wind_001.wav        # 바람 소리
│   └── rain_001.wav        # 비 소리
├── mechanical/             # 기계 소음
│   ├── fan_001.wav         # 팬 소리
│   └── ac_001.wav          # 에어컨 소리
└── background/             # 배경 소음
    ├── chatter_001.wav     # 대화 소리
    └── music_001.wav       # 음악 소리
```

#### 최소 요구사항:
- 최소 **5개 이상**의 다양한 소음 파일
- 각기 다른 유형의 소음 (환경적, 기계적, 인공적 등)

### 3. 오디오 파일 형식 요구사항

모든 오디오 파일은 다음 조건을 만족해야 합니다:

- **형식**: .wav (PCM 형식)
- **비트 깊이**: 16-bit 권장
- **샘플링 레이트**: 22050Hz 또는 44100Hz 권장
- **채널**: 모노 또는 스테레오 (시스템에서 자동으로 모노로 변환)
- **길이**: 1-10초 권장

### 4. 데이터 검증

데이터 배치 후 다음을 확인해주세요:

```bash
# 디렉토리 구조 확인
ls -la data/raw/
ls -la data/noise/

# 파일 개수 확인
# 훈련 데이터
find data/raw/train/watermelon_A/ -name "*.wav" | wc -l
find data/raw/train/watermelon_B/ -name "*.wav" | wc -l  
find data/raw/train/watermelon_C/ -name "*.wav" | wc -l

# 검증 데이터
find data/raw/validation/watermelon_A/ -name "*.wav" | wc -l
find data/raw/validation/watermelon_B/ -name "*.wav" | wc -l  
find data/raw/validation/watermelon_C/ -name "*.wav" | wc -l

# 테스트 데이터
find data/raw/test/watermelon_A/ -name "*.wav" | wc -l
find data/raw/test/watermelon_B/ -name "*.wav" | wc -l  
find data/raw/test/watermelon_C/ -name "*.wav" | wc -l

# 소음 파일
find data/noise/environmental/retail/homeplus/ -name "*.wav" | wc -l
find data/noise/environmental/retail/emart/ -name "*.wav" | wc -l
find data/noise/mechanical/ -name "*.wav" | wc -l
find data/noise/background/ -name "*.wav" | wc -l
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd wm_mlt_models

# 가상환경 생성 (권장)
python -m venv watermelon_env
source watermelon_env/bin/activate  # Linux/Mac
# 또는 watermelon_env\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터를 올바른 구조로 배치해주세요:

```
data/raw/
├── train/          # 훈련 데이터 (70%)
│   ├── watermelon_A/
│   ├── watermelon_B/
│   └── watermelon_C/
├── validation/     # 검증 데이터 (20%)
│   ├── watermelon_A/
│   ├── watermelon_B/
│   └── watermelon_C/
└── test/          # 테스트 데이터 (10%)
    ├── watermelon_A/
    ├── watermelon_B/
    └── watermelon_C/

data/noise/        # 소음 파일들 (선택사항)
├── environmental/retail/
│   ├── homeplus/
│   └── emart/
├── mechanical/
└── background/
```

자세한 데이터 배치 방법은 [DATA_PLACEMENT_GUIDE.md](DATA_PLACEMENT_GUIDE.md)를 참조하세요.

### 3. 실행

```bash
# 기본 실행 (모든 단계 자동 실행)
python main.py

# 소음 증강 없이 실행 (소음 파일이 없는 경우)
python main.py --skip-augmentation

# 5-fold 교차 검증으로 실행 (기본값)
python main.py --cv-folds 5

# Core ML 변환 제외하고 실행
python main.py --no-coreml

# 상태 확인
python main.py --status

# 중단된 실행 재개
python main.py --resume

# 도움말
python main.py --help
```

## 📊 출력 결과

실행 완료 후 다음 위치에서 결과를 확인할 수 있습니다:

### 모델 파일
- **Pickle 모델**: `data/models/pickle/svm_model.pkl`, `data/models/pickle/random_forest_model.pkl`
- **Core ML 모델**: `data/models/coreml/svm_model.mlmodel`, `data/models/coreml/random_forest_model.mlmodel`
- **모델 메타데이터**: `data/models/artifacts/model_metadata.json`

### 보고서 및 로그
- **성능 평가 보고서**: `results/evaluation_report.json`
- **성능 모니터링**: `results/performance_report.json`
- **데이터 무결성 보고서**: `results/integrity_report.json`
- **실행 로그**: `logs/watermelon_classifier_YYYYMMDD.log`

### 처리된 데이터
- **증강된 훈련 데이터**: `data/processed/augmented/`
- **추출된 특징**: `data/processed/features/`
- **데이터 분할 정보**: `data/processed/splits/`

## 📚 문서

- **[사용 예제](docs/USAGE_EXAMPLES.md)**: 다양한 사용 사례와 코드 예제
- **[API 참조](docs/API_REFERENCE.md)**: 모든 클래스와 함수에 대한 상세 문서
- **[Core ML 사용법](docs/COREML_USAGE.md)**: iOS/macOS에서 모델 사용 방법
- **[데이터 배치 가이드](DATA_PLACEMENT_GUIDE.md)**: 데이터 구조 및 배치 방법

## 🔧 고급 사용법

### 프로그래밍 방식 사용

```python
from main import WatermelonClassificationPipeline
from config import Config

# 사용자 정의 설정으로 실행
config = Config(
    cv_folds=10,
    snr_levels=[-10, -5, 0, 5, 10],
    augmentation_factor=6
)

pipeline = WatermelonClassificationPipeline(
    config=config,
    enable_performance_monitoring=True
)
results = pipeline.run()
```

### 개별 구성요소 사용

```python
from src.audio.feature_extraction import AudioFeatureExtractor
from src.ml.training import ModelTrainer

# 특징 추출
extractor = AudioFeatureExtractor()
features = extractor.extract_features("audio.wav")

# 모델 훈련
trainer = ModelTrainer()
results = trainer.train_with_cv(X_train, y_train)
```

### 실시간 예측

```python
from src.audio.feature_extraction import AudioFeatureExtractor
from src.ml.training import ModelTrainer

# 모델 로드
extractor = AudioFeatureExtractor()
trainer = ModelTrainer()
model = trainer.load_model("data/models/pickle/svm_model.pkl")

# 예측
features = extractor.extract_features("new_watermelon.wav")
prediction = model.predict(features.reshape(1, -1))[0]
confidence = max(model.predict_proba(features.reshape(1, -1))[0])

print(f"예측: {prediction}, 신뢰도: {confidence:.4f}")
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. 설치 관련 문제

**문제**: `librosa` 설치 실패
```bash
# 해결책: 시스템 종속성 설치
# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile

# Windows: conda 사용 권장
conda install -c conda-forge librosa
```

**문제**: `coremltools` 설치 실패 (Apple Silicon Mac)
```bash
# 해결책: 네이티브 버전 설치
pip install --upgrade coremltools
```

#### 2. 데이터 관련 문제

**문제**: "No audio files found" 오류
- 데이터 디렉토리 구조가 올바른지 확인
- 파일 확장자가 `.wav`인지 확인
- 파일 권한이 읽기 가능한지 확인

**문제**: "Insufficient data" 경고
- 각 클래스당 최소 20개 파일 필요 (훈련용)
- 검증/테스트 데이터도 충분한지 확인

**문제**: 오디오 로딩 오류
```python
# 파일 검증 방법
from src.utils.file_utils import AudioFileUtils
validation = AudioFileUtils.validate_audio_file("audio.wav")
print(validation)
```

#### 3. 메모리 관련 문제

**문제**: 메모리 부족 오류
```bash
# 해결책: 배치 크기 줄이기
python main.py --no-performance-monitoring
```

**문제**: 디스크 공간 부족
- 임시 증강 파일 정리: `data/processed/augmented/` 디렉토리 확인
- 로그 파일 정리: `logs/` 디렉토리 확인

#### 4. 성능 관련 문제

**문제**: 훈련이 너무 느림
```bash
# 해결책: 교차 검증 폴드 수 줄이기
python main.py --cv-folds 3

# 또는 증강 생략
python main.py --skip-augmentation
```

### 디버깅 팁

1. **상세 로그 확인**: `logs/` 디렉토리의 최신 로그 파일 확인
2. **데이터 검증**: `python -c "from src.utils.data_integrity import DataIntegrityChecker; checker = DataIntegrityChecker(); print(checker.get_summary_report())"`
3. **시스템 리소스 확인**: `python -c "from src.utils.performance_monitor import PerformanceMonitor; monitor = PerformanceMonitor(); print(monitor.get_system_info())"`

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🙏 감사의 말

- **librosa**: 오디오 신호 처리 라이브러리
- **scikit-learn**: 머신러닝 알고리즘 및 유틸리티
- **coremltools**: Core ML 모델 변환 도구

## 📞 지원 및 문의

문제가 발생하거나 질문이 있으시면:
1. [Issues](../../issues) 탭에서 기존 이슈 확인
2. 새로운 이슈 생성 (버그 리포트 또는 기능 요청)
3. 프로젝트 문서 참조