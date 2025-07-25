# 증강 데이터 사용 가이드

이 문서는 `--use-existing-augmented` 옵션을 사용하여 기존 증강 데이터를 재사용하는 방법을 설명합니다.

## 개요

`--use-existing-augmented` 옵션은 이미 생성된 증강 데이터를 재사용하여 학습 시간을 단축하면서도 증강의 이점을 유지할 수 있게 해줍니다.

## 증강 데이터 위치

증강된 오디오 파일들은 다음 경로에 저장됩니다:
```
data/processed/augmented/
├── watermelon_A/
│   ├── original_file_noise_homeplus_snr+0dB.wav
│   ├── original_file_noise_homeplus_snr+5dB.wav
│   └── ...
├── watermelon_B/
│   └── ...
└── watermelon_C/
    └── ...
```

## 사용 시나리오

### 시나리오 1: 최초 실행 (증강 데이터 생성)
```bash
# 처음 실행 시 증강 데이터 생성
python main.py
```
- 새로운 증강 데이터 생성
- `data/processed/augmented/` 디렉토리에 저장
- 시간이 오래 걸림 (데이터 양에 따라 5-30분)

### 시나리오 2: 동일한 증강 데이터로 재학습
```bash
# 기존 증강 데이터 재사용
python main.py --skip-augmentation --use-existing-augmented
```
- 새로운 증강 생성하지 않음
- 기존 증강 파일 자동 탐색 및 로드
- 빠른 실행 (증강 단계 스킵)

### 시나리오 3: 빠른 프로토타이핑 (원본만)
```bash
# 증강 없이 원본 데이터만 사용
python main.py --skip-augmentation
```
- 가장 빠른 실행
- 증강 효과 없음

### 시나리오 4: 증강 데이터 재생성
```bash
# 기존 증강 무시하고 새로 생성
python main.py
```
- 기존 증강 파일 덮어쓰기
- 다른 SNR 레벨이나 노이즈 파일로 재생성 시 유용

## 옵션 조합 매트릭스

| `--skip-augmentation` | `--use-existing-augmented` | 동작 | 사용 시점 |
|----------------------|---------------------------|------|-----------|
| ❌ (기본값) | ❌ (기본값) | 새 증강 생성 | 최초 실행 |
| ✅ | ❌ | 원본만 사용 | 빠른 테스트 |
| ✅ | ✅ | 기존 증강 재사용 | **권장** |
| ❌ | ✅ | 새 증강 생성 | 증강 재생성 |

## 실행 예시

### 전체 워크플로우
```bash
# 1. 최초 실행 - 증강 데이터 생성
python main.py
# 출력: "증강 완료: watermelon_A, 30개 파일 생성"

# 2. 동일 증강으로 다른 하이퍼파라미터 테스트
python main.py --skip-augmentation --use-existing-augmented --cv-folds 10
# 출력: "기존 증강 데이터를 로드합니다."
# 출력: "30개 증강 파일 발견"

# 3. 원본만으로 빠른 테스트
python main.py --skip-augmentation
# 출력: "증강 건너뜀 - 원본 데이터만 사용"
```

## 증강 데이터 관리

### 증강 데이터 확인
```bash
# 생성된 증강 파일 개수 확인
find data/processed/augmented -name "*.wav" | wc -l

# 클래스별 증강 파일 확인
ls -la data/processed/augmented/watermelon_A/
```

### 증강 데이터 삭제
```bash
# 모든 증강 데이터 삭제 (재생성 필요 시)
rm -rf data/processed/augmented/*
```

### 디스크 사용량 확인
```bash
# 증강 데이터 크기 확인
du -sh data/processed/augmented/
```

## 주의사항

1. **일관성 유지**: 증강 파라미터(SNR 레벨, 노이즈 파일)가 변경되면 증강 데이터를 재생성해야 합니다.

2. **디스크 공간**: 증강 데이터는 원본의 4-5배 공간을 차지할 수 있습니다.
   - 원본: 100MB → 증강 후: 500MB

3. **파일 무결성**: 증강 파일이 손상되거나 삭제된 경우, 재생성이 필요합니다.

4. **버전 관리**: 증강 파일은 `.gitignore`에 포함시켜 저장소 크기를 관리하세요.

## 성능 비교

| 모드 | 실행 시간 | 메모리 사용 | 모델 성능 |
|------|-----------|-------------|-----------|
| 새 증강 생성 | ~15분 | 높음 | 최상 |
| 기존 증강 재사용 | ~3분 | 중간 | 최상 |
| 원본만 사용 | ~1분 | 낮음 | 기본 |

## 문제 해결

### "증강 디렉토리 없음" 경고
```bash
# 증강 데이터를 한 번도 생성하지 않은 경우
python main.py  # 먼저 증강 생성
```

### "0개 증강 파일 발견"
```bash
# 증강 파일이 삭제되었거나 경로가 잘못된 경우
# 증강 데이터 재생성
python main.py
```

### 메모리 부족
```bash
# 증강 없이 실행하거나
python main.py --skip-augmentation

# 또는 배치 크기 조정 (코드 수정 필요)
```

## 고급 사용법

### 특정 클래스만 증강 사용
현재는 전체 클래스에 대해 동일하게 적용되지만, 코드를 수정하여 클래스별 제어가 가능합니다.

### 증강 파라미터 변경
`config/config.py`에서 SNR 레벨과 증강 배수를 조정할 수 있습니다:
```python
snr_levels: List[float] = [-5, 0, 5, 10]
augmentation_factor: int = 4
```

## 관련 문서

- [메인 파이프라인 가이드](USAGE.md)
- [단일 모델 학습 가이드](SINGLE_MODEL_TRAINING_GUIDE.md)
- [데이터 준비 가이드](data_preparation_guide.md)