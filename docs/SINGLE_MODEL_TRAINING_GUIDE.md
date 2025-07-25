# 단일 모델 학습 가이드

이 문서는 `train_single_model.py` 스크립트를 사용하여 하나의 모델만 학습시키는 방법을 설명합니다.

## 개요

`train_single_model.py`는 전체 파이프라인을 실행하지 않고 SVM 또는 Random Forest 중 하나의 모델만 빠르게 학습하고 평가할 수 있는 독립적인 스크립트입니다.

## 주요 특징

- **빠른 실행**: 하나의 모델만 학습하므로 실행 시간이 단축됩니다
- **유연한 옵션**: 데이터 증강 포함/제외 선택 가능
- **즉각적인 결과**: 학습 후 바로 성능 평가 결과 확인
- **선택적 저장**: 모델 저장 여부를 대화형으로 선택

## 사용법

### 기본 사용법

```bash
# SVM 모델만 학습 (기본값, 데이터 증강 없이)
python train_single_model.py

# Random Forest 모델만 학습
python train_single_model.py --model random_forest
```

### 옵션 설명

#### `--model` (모델 선택)
- **설명**: 학습할 모델 타입을 선택합니다
- **선택지**: `svm`, `random_forest`
- **기본값**: `svm`

```bash
# SVM 학습
python train_single_model.py --model svm

# Random Forest 학습
python train_single_model.py --model random_forest
```

#### `--with-augmentation` (데이터 증강 포함)
- **설명**: 데이터 증강을 포함하여 학습합니다
- **기본값**: 증강 없음 (빠른 실행을 위해)

```bash
# 데이터 증강 포함하여 SVM 학습
python train_single_model.py --model svm --with-augmentation

# 데이터 증강 포함하여 Random Forest 학습
python train_single_model.py --model random_forest --with-augmentation
```

#### `--use-existing-augmented` (기존 증강 데이터 사용)
- **설명**: 기존에 생성된 증강 데이터를 재사용합니다
- **기본값**: 사용 안함
- **장점**: 증강 생성 시간 절약 + 증강 효과 유지

```bash
# 기존 증강 데이터로 SVM 학습 (권장)
python train_single_model.py --model svm --use-existing-augmented

# 기존 증강 데이터로 Random Forest 학습
python train_single_model.py --model random_forest --use-existing-augmented
```

## 실행 과정

### 1. 데이터 로딩
- 훈련/검증/테스트 데이터 로드
- 특징 추출 수행
- 데이터 증강 (선택사항)

### 2. 모델 훈련
- GridSearchCV를 통한 하이퍼파라미터 최적화
- 5-fold 교차 검증
- 최적 파라미터 선택

### 3. 모델 평가
- 테스트 세트에서 성능 평가
- 정확도, F1-score, Precision, Recall 계산
- 클래스별 성능 분석

### 4. 결과 출력
- 최적 하이퍼파라미터
- 전체 성능 지표
- 클래스별 상세 성능
- Random Forest의 경우 특징 중요도

### 5. 모델 저장 (선택사항)
- 대화형 프롬프트로 저장 여부 결정
- Pickle 형식으로 저장

## 출력 예시

```
🎯 SVM 모델만 학습을 시작합니다.
============================================================

📊 1단계: 데이터 로딩 중...
✅ 데이터 로딩 완료:
  훈련 데이터: 120개
  검증 데이터: 40개
  테스트 데이터: 40개

🤖 2단계: SVM 모델 훈련 중...
✅ 모델 훈련 완료:
  최적 점수: 0.8542
  최적 파라미터: {'C': 10, 'gamma': 0.01}
  훈련 시간: 15.32초

📈 3단계: 모델 평가 중...
✅ 모델 평가 완료:
  정확도: 0.8750
  F1-score (macro): 0.8695
  정밀도 (macro): 0.8811
  재현율 (macro): 0.8750

📊 클래스별 성능:
  watermelon_A:
    정밀도: 0.9231
    재현율: 0.9000
    F1-score: 0.9114
  watermelon_B:
    정밀도: 0.8333
    재현율: 0.8571
    F1-score: 0.8451
  watermelon_C:
    정밀도: 0.8667
    재현율: 0.8667
    F1-score: 0.8667

💾 모델을 저장하시겠습니까? (y/N): 

⏱️  전체 실행 시간: 42.15초
```

## 사용 시나리오

### 1. 빠른 프로토타이핑
```bash
# 데이터 증강 없이 빠르게 SVM 테스트
python train_single_model.py --model svm
```

### 2. 모델 비교를 위한 개별 실행
```bash
# 기존 증강 데이터로 SVM 실행 후 결과 기록
python train_single_model.py --model svm --use-existing-augmented > svm_results.txt

# 기존 증강 데이터로 Random Forest 실행 후 결과 기록
python train_single_model.py --model random_forest --use-existing-augmented > rf_results.txt
```

### 3. 전체 데이터로 최종 모델 학습
```bash
# 기존 증강 데이터로 최종 모델 학습 (권장)
python train_single_model.py --model svm --use-existing-augmented

# 또는 새로운 증강 데이터 생성
python train_single_model.py --model svm --with-augmentation
```

### 4. 증강 데이터 워크플로우
```bash
# 1단계: 최초 증강 데이터 생성
python train_single_model.py --model svm --with-augmentation

# 2단계: 생성된 증강 데이터 재사용 (빠른 실행)
python train_single_model.py --model random_forest --use-existing-augmented

# 3단계: 다른 설정으로 동일 증강 데이터 재사용
python train_single_model.py --model svm --use-existing-augmented
```

## 주의사항

1. **데이터 요구사항**: `data/raw` 디렉토리에 훈련/검증/테스트 데이터가 있어야 합니다
2. **메모리 사용**: 데이터 증강을 포함하면 메모리 사용량이 증가합니다
3. **실행 시간**: 
   - 증강 없이: 약 30초~1분
   - 증강 포함: 약 2~5분 (데이터 양에 따라 다름)

## 문제 해결

### "모듈을 찾을 수 없음" 오류
```bash
# 프로젝트 루트에서 실행하는지 확인
cd /path/to/wm_mlt_models
python train_single_model.py
```

### 메모리 부족 오류
```bash
# 데이터 증강 없이 실행
python train_single_model.py --model svm
```

### 긴 실행 시간
- 데이터 증강을 제외하고 실행
- 더 작은 교차 검증 폴드 수 사용 (코드 수정 필요)

## 다음 단계

1. 단일 모델 학습 후 만족스러운 결과를 얻었다면:
   - `python main.py`로 전체 파이프라인 실행
   - Core ML 변환을 위해 모델 저장

2. 성능을 개선하고 싶다면:
   - 데이터 증강 포함하여 재학습
   - 다른 모델 타입 시도
   - 하이퍼파라미터 범위 조정 (코드 수정 필요)

## 관련 문서

- [메인 파이프라인 가이드](USAGE.md)
- [데이터 준비 가이드](data_preparation_guide.md)
- [Core ML 사용 가이드](COREML_USAGE.md)