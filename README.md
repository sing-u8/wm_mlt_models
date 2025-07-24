# 수박 소리 분류 머신러닝 시스템

이 프로젝트는 수박의 숙성도에 따른 소리를 분류하는 파이썬 기반 머신러닝 파이프라인입니다. 시스템은 librosa를 활용한 오디오 특징 추출, 소음 증강 기법, scikit-learn 기반 모델 훈련을 통해 강건한 분류 성능을 제공합니다.

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

## 실행 방법

데이터 준비가 완료되면 다음 명령으로 파이프라인을 실행할 수 있습니다:

```bash
# 의존성 설치
pip install -r requirements.txt

# 전체 파이프라인 실행
python src/main.py

# 또는 단계별 실행
python src/main.py --step data_split
python src/main.py --step augmentation  
python src/main.py --step training
python src/main.py --step evaluation
```

## 출력 결과

실행 완료 후 다음 위치에서 결과를 확인할 수 있습니다:

- **훈련된 모델**: `data/models/pickle/` (pickle 형식)
- **Core ML 모델**: `data/models/coreml/` (iOS/macOS용)
- **성능 보고서**: `results/evaluation_report.html`
- **로그 파일**: `logs/pipeline.log`

## 문제 해결

### 일반적인 문제들:

1. **"No audio files found" 오류**
   - 데이터 디렉토리 구조가 올바른지 확인
   - 파일 확장자가 .wav인지 확인

2. **"Insufficient data" 경고**
   - 각 클래스당 최소 20개 파일이 있는지 확인
   - 소음 파일이 최소 5개 있는지 확인

3. **오디오 로딩 오류**
   - 파일이 손상되지 않았는지 확인
   - 지원되는 오디오 형식(.wav)인지 확인

더 자세한 정보는 프로젝트 문서를 참조해주세요.