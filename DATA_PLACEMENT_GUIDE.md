# 데이터 배치 가이드

## 개요

이 가이드는 수박 소리 분류 프로젝트에서 오디오 데이터를 올바르게 배치하는 방법을 설명합니다.

## 데이터 디렉토리 구조

프로젝트의 `data/` 디렉토리는 다음과 같이 구성되어 있습니다:

```
data/
├── raw/                        # 원본 오디오 파일들 (이미 분할됨)
│   ├── train/                 # 훈련용 데이터 (70%)
│   │   ├── watermelon_A/      # 수박 A 유형 (예: 덜 익은 수박)
│   │   ├── watermelon_B/      # 수박 B 유형 (예: 적당히 익은 수박)
│   │   └── watermelon_C/      # 수박 C 유형 (예: 잘 익은 수박)
│   ├── validation/            # 검증용 데이터 (20%)
│   │   ├── watermelon_A/
│   │   ├── watermelon_B/
│   │   └── watermelon_C/
│   └── test/                  # 테스트용 데이터 (10%)
│       ├── watermelon_A/
│       ├── watermelon_B/
│       └── watermelon_C/
├── noise/                      # 소음 파일들
│   ├── environmental/         # 환경 소음
│   │   └── retail/            # 소매점/마트 환경 소음
│   │       ├── homeplus/      # 홈플러스 환경 소음
│   │       └── emart/         # 이마트 환경 소음
│   ├── mechanical/            # 기계 소음 (선택사항, 파일 없어도 무관)
│   └── background/            # 배경 소음 (선택사항, 파일 없어도 무관)
├── processed/                  # 처리된 데이터 (자동 생성)
│   ├── augmented/             # 증강된 훈련 데이터만 저장
│   │   ├── watermelon_A/
│   │   ├── watermelon_B/
│   │   └── watermelon_C/
│   ├── features/              # 추출된 특징 파일
│   └── splits/                # 데이터 분할 정보 (메타데이터)
└── models/                     # 훈련된 모델 (자동 생성)
    ├── artifacts/             # 모델 메타데이터
    ├── pickle/                # Pickle 형식 모델
    └── coreml/                # Core ML 형식 모델
```

## 데이터 배치 방법

### 1. 원본 수박 소리 파일 배치

데이터는 이미 train/validation/test로 분할되어 배치되어야 합니다:

#### 훈련 데이터 (70%)

- **watermelon_A**: `data/raw/train/watermelon_A/` 디렉토리에 배치
- **watermelon_B**: `data/raw/train/watermelon_B/` 디렉토리에 배치
- **watermelon_C**: `data/raw/train/watermelon_C/` 디렉토리에 배치

#### 검증 데이터 (20%)

- **watermelon_A**: `data/raw/validation/watermelon_A/` 디렉토리에 배치
- **watermelon_B**: `data/raw/validation/watermelon_B/` 디렉토리에 배치
- **watermelon_C**: `data/raw/validation/watermelon_C/` 디렉토리에 배치

#### 테스트 데이터 (10%)

- **watermelon_A**: `data/raw/test/watermelon_A/` 디렉토리에 배치
- **watermelon_B**: `data/raw/test/watermelon_B/` 디렉토리에 배치
- **watermelon_C**: `data/raw/test/watermelon_C/` 디렉토리에 배치

#### 파일 명명 규칙

권장하는 파일 명명 규칙을 따르세요:

- 형식: `{class_name}_{sample_id}.wav`
- 예시:
  - `watermelon_A_001.wav`
  - `watermelon_A_002.wav`
  - `watermelon_B_001.wav`
  - `watermelon_C_001.wav`

### 2. 소음 파일 배치

소음 파일들을 유형별로 분류하여 배치하세요:

- **소매점 환경 소음**: `data/noise/environmental/retail/`
  - 홈플러스 환경 소음: `data/noise/environmental/retail/homeplus/`
    - 예: `homeplus_ambient_001.wav`, `homeplus_crowd_001.wav`
  - 이마트 환경 소음: `data/noise/environmental/retail/emart/`
    - 예: `emart_ambient_001.wav`, `emart_crowd_001.wav`
- **기계 소음**: `data/noise/mechanical/` (선택사항)
  - 에어컨, 팬, 기계 작동음 등
  - 예: `mechanical_fan_001.wav`
- **배경 소음**: `data/noise/background/` (선택사항)
  - 대화 소리, 음악, TV 소리 등
  - 예: `background_chatter_001.wav`

**중요**: mechanical/과 background/ 폴더에 소음 파일이 없어도 시스템은 정상적으로 동작합니다. 시스템은 사용 가능한 소음 파일만을 자동으로 감지하여 사용합니다.

### 3. 파일 요구사항

#### 최소 데이터 요구사항

- **훈련 데이터**: 각 클래스당 **최소 14개 이상**의 파일 (전체의 70%)
- **검증 데이터**: 각 클래스당 **최소 4개 이상**의 파일 (전체의 20%)
- **테스트 데이터**: 각 클래스당 **최소 2개 이상**의 파일 (전체의 10%)
- **소음 파일**: **최소 5개 이상**의 다양한 소음 파일

#### 오디오 형식 요구사항

- **파일 형식**: `.wav` (필수)
- **인코딩**: 16-bit PCM 권장
- **샘플링 레이트**: 22050Hz 또는 44100Hz 권장
- **채널**: 모노(1채널) 또는 스테레오(2채널) 모두 지원

## 데이터 검증

올바르게 배치된 데이터는 다음 명령어로 확인할 수 있습니다:

```python
from config import DEFAULT_CONFIG

# 설정 확인
config = DEFAULT_CONFIG
print("클래스 디렉토리:", config.get_class_directories())
print("소음 디렉토리:", config.get_noise_subdirectories())
```

## 예시 데이터 구조

올바르게 구성된 데이터 구조의 예시:

```
data/raw/
├── train/
│   ├── watermelon_A/
│   │   ├── watermelon_A_001.wav
│   │   ├── watermelon_A_002.wav
│   │   └── ... (최소 14개 파일)
│   ├── watermelon_B/
│   │   ├── watermelon_B_001.wav
│   │   └── ... (최소 14개 파일)
│   └── watermelon_C/
│       ├── watermelon_C_001.wav
│       └── ... (최소 14개 파일)
├── validation/
│   ├── watermelon_A/
│   │   ├── watermelon_A_val_001.wav
│   │   └── ... (최소 4개 파일)
│   ├── watermelon_B/
│   │   └── ... (최소 4개 파일)
│   └── watermelon_C/
│       └── ... (최소 4개 파일)
└── test/
    ├── watermelon_A/
    │   ├── watermelon_A_test_001.wav
    │   └── ... (최소 2개 파일)
    ├── watermelon_B/
    │   └── ... (최소 2개 파일)
    └── watermelon_C/
        └── ... (최소 2개 파일)

data/noise/
├── environmental/
│   └── retail/
│       ├── homeplus/
│       │   ├── homeplus_ambient_001.wav
│       │   ├── homeplus_crowd_001.wav
│       │   └── homeplus_checkout_001.wav
│       └── emart/
│           ├── emart_ambient_001.wav
│           ├── emart_crowd_001.wav
│           └── emart_announcement_001.wav
├── mechanical/
│   ├── fan_ceiling_001.wav
│   ├── ac_running_001.wav
│   └── motor_hum_001.wav
└── background/
    ├── conversation_cafe_001.wav
    ├── music_ambient_001.wav
    └── tv_background_001.wav
```

## 주의사항

1. **파일 형식**: 반드시 `.wav` 형식을 사용하세요
2. **파일 크기**: 너무 큰 파일(>100MB)은 처리 시간이 오래 걸릴 수 있습니다
3. **품질**: 손상되거나 무음인 파일은 자동으로 감지되어 제외됩니다
4. **클래스 균형**: 각 클래스별로 비슷한 수의 샘플을 준비하는 것이 좋습니다

## 문제 해결

### 자주 발생하는 문제들

1. **"파일을 찾을 수 없습니다" 오류**

   - 파일이 올바른 디렉토리에 있는지 확인
   - 파일 경로에 한글이나 특수문자가 있는지 확인

2. **"지원되지 않는 오디오 형식" 오류**

   - 파일이 `.wav` 형식인지 확인
   - 오디오 인코딩이 올바른지 확인

3. **"데이터가 부족합니다" 경고**
   - 훈련 데이터: 각 클래스별로 최소 14개 파일이 있는지 확인
   - 검증 데이터: 각 클래스별로 최소 4개 파일이 있는지 확인
   - 테스트 데이터: 각 클래스별로 최소 2개 파일이 있는지 확인
   - 소음 파일이 최소 5개 있는지 확인

더 자세한 도움이 필요하면 프로젝트의 README.md 파일을 참조하세요.
