# 수박 소리 분류 시스템 사용 예제

이 문서는 수박 소리 분류 시스템의 다양한 구성요소를 사용하는 방법을 예제와 함께 설명합니다.

## 목차

1. [기본 파이프라인 실행](#기본-파이프라인-실행)
2. [특징 추출 사용법](#특징-추출-사용법)
3. [데이터 증강 예제](#데이터-증강-예제)
4. [모델 훈련 및 평가](#모델-훈련-및-평가)
5. [모델 변환 및 배포](#모델-변환-및-배포)
6. [실시간 예측 사용](#실시간-예측-사용)
7. [유틸리티 함수 활용](#유틸리티-함수-활용)
8. [성능 모니터링](#성능-모니터링)
9. [고급 사용법](#고급-사용법)

---

## 기본 파이프라인 실행

### 1. 전체 파이프라인 실행

```python
#!/usr/bin/env python3
"""
기본 파이프라인 실행 예제
"""

from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from main import WatermelonClassificationPipeline
from config import DEFAULT_CONFIG

def run_basic_pipeline():
    """기본적인 전체 파이프라인 실행"""
    
    # 파이프라인 초기화
    pipeline = WatermelonClassificationPipeline(
        config=DEFAULT_CONFIG,
        checkpoint_dir="checkpoints",
        enable_integrity_checks=True,
        enable_performance_monitoring=True
    )
    
    # 파이프라인 실행
    results = pipeline.run_complete_pipeline()
    
    print("파이프라인 실행 완료!")
    print(f"최종 결과: {results}")
    
    return results

if __name__ == "__main__":
    run_basic_pipeline()
```

### 2. 명령줄에서 실행

```bash
# 기본 실행
python main.py

# 증강 없이 실행
python main.py --skip-augmentation

# Core ML 변환 없이 실행
python main.py --no-coreml

# 3-fold 교차 검증으로 실행
python main.py --cv-folds 3

# 중단된 파이프라인 재개
python main.py --resume

# 파이프라인 상태 확인
python main.py --status

# 무결성 검사 없이 실행
python main.py --no-integrity-checks

# 성능 모니터링 없이 실행
python main.py --no-performance-monitoring
```

---

## 특징 추출 사용법

### 1. 단일 오디오 파일 특징 추출

```python
"""
오디오 특징 추출 예제
"""

from src.audio.feature_extraction import AudioFeatureExtractor
from pathlib import Path

def extract_single_file_features():
    """단일 파일에서 특징 추출"""
    
    extractor = AudioFeatureExtractor()
    
    # 오디오 파일 경로
    audio_path = "data/raw/train/ripe/sample_001.wav"
    
    # 특징 추출
    features = extractor.extract_features(audio_path)
    
    print(f"추출된 특징 벡터 크기: {features.shape}")
    print(f"특징 벡터: {features[:5]}...")  # 처음 5개 값만 출력
    
    return features

def extract_batch_features():
    """여러 파일에서 배치 특징 추출"""
    
    extractor = AudioFeatureExtractor()
    
    # 오디오 파일들 찾기
    audio_files = list(Path("data/raw/train/ripe").glob("*.wav"))
    
    all_features = []
    for audio_file in audio_files[:5]:  # 처음 5개 파일만
        try:
            features = extractor.extract_features(str(audio_file))
            all_features.append(features)
            print(f"✅ {audio_file.name}: {features.shape}")
        except Exception as e:
            print(f"❌ {audio_file.name}: {e}")
    
    return all_features

if __name__ == "__main__":
    # 단일 파일 특징 추출
    features = extract_single_file_features()
    
    # 배치 특징 추출
    batch_features = extract_batch_features()
```

### 2. 특징 분석 및 시각화

```python
"""
특징 분석 및 시각화 예제
"""

import numpy as np
from src.utils.file_utils import VisualizationUtils, ArrayUtils

def analyze_features():
    """추출된 특징 분석"""
    
    # 가상의 특징 데이터 생성 (실제로는 extract_features에서 가져옴)
    np.random.seed(42)
    X = np.random.randn(100, 30)  # 100개 샘플, 30차원 특징
    y = np.random.choice([0, 1, 2], size=100)  # 3개 클래스
    
    # 특징 정규화
    X_normalized, norm_params = ArrayUtils.normalize_features(X, method='standard')
    print(f"정규화 파라미터: {norm_params['method']}")
    
    # 특징 중요도 계산
    importance = ArrayUtils.compute_feature_importance(X_normalized, y)
    print(f"상위 5개 중요 특징: {list(importance.items())[:5]}")
    
    # 시각화
    feature_names = [f'MFCC_{i}' if i < 13 else 
                    f'Mel_Stat_{i-13}' if i < 15 else
                    f'Spectral_{i-15}' if i < 18 else
                    f'Chroma_{i-18}' for i in range(30)]
    
    # 특징 분포 시각화
    VisualizationUtils.plot_feature_distribution(
        X_normalized, 
        feature_names=feature_names,
        output_path="results/feature_distribution.png"
    )
    
    # 상관관계 매트릭스
    VisualizationUtils.plot_correlation_matrix(
        X_normalized,
        feature_names=feature_names,
        output_path="results/correlation_matrix.png"
    )
    
    # 클래스 분포
    VisualizationUtils.plot_class_distribution(
        y,
        class_names=['Unripe', 'Ripe', 'Overripe'],
        output_path="results/class_distribution.png"
    )
    
    # 특징 중요도 시각화
    VisualizationUtils.plot_feature_importance(
        importance,
        top_n=15,
        output_path="results/feature_importance.png"
    )

if __name__ == "__main__":
    analyze_features()
```

---

## 데이터 증강 예제

### 1. 소음 증강 사용법

```python
"""
데이터 증강 예제
"""

from src.data.augmentation import BatchAugmentor
from config import DEFAULT_CONFIG

def augment_training_data():
    """훈련 데이터 증강"""
    
    augmentor = BatchAugmentor(config=DEFAULT_CONFIG)
    
    # 단일 클래스 디렉토리 증강
    input_dir = "data/raw/train/ripe"
    output_dir = "data/augmented/train/ripe"
    
    augmented_files = augmentor.augment_class_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_multiplier=3  # 3배 증강
    )
    
    print(f"증강된 파일 수: {len(augmented_files)}")
    for file_info in augmented_files[:5]:  # 처음 5개만 출력
        print(f"  {file_info['output_path']} (SNR: {file_info['snr_db']}dB)")
    
    return augmented_files

def custom_snr_augmentation():
    """커스텀 SNR 레벨로 증강"""
    
    from src.data.augmentation import AudioAugmentor
    
    augmentor = AudioAugmentor(config=DEFAULT_CONFIG)
    
    # 원본 오디오와 소음 파일
    clean_audio_path = "data/raw/train/ripe/sample_001.wav"
    noise_audio_path = "data/noise/environmental/retail/homeplus/ambient_001.wav"
    
    # 다양한 SNR 레벨로 증강
    snr_levels = [5, 10, 15, 20]
    
    for snr in snr_levels:
        output_path = f"data/augmented/custom/sample_001_snr_{snr}db.wav"
        
        success = augmentor.augment_noise(
            clean_audio_path=clean_audio_path,
            noise_audio_path=noise_audio_path,
            output_path=output_path,
            snr_db=snr
        )
        
        if success:
            print(f"✅ SNR {snr}dB 증강 완료: {output_path}")
        else:
            print(f"❌ SNR {snr}dB 증강 실패")

if __name__ == "__main__":
    # 배치 증강
    augment_training_data()
    
    # 커스텀 SNR 증강
    custom_snr_augmentation()
```

---

## 모델 훈련 및 평가

### 1. 모델 훈련

```python
"""
모델 훈련 예제
"""

import numpy as np
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from config import DEFAULT_CONFIG

def train_models():
    """SVM과 Random Forest 모델 훈련"""
    
    # 가상의 훈련 데이터 (실제로는 데이터 파이프라인에서 가져옴)
    np.random.seed(42)
    X_train = np.random.randn(200, 30)
    y_train = np.random.choice([0, 1, 2], size=200)
    X_val = np.random.randn(50, 30)
    y_val = np.random.choice([0, 1, 2], size=50)
    
    # 모델 트레이너 초기화
    trainer = ModelTrainer(config=DEFAULT_CONFIG)
    
    # 교차 검증으로 훈련
    training_results = trainer.train_with_cv(
        X_train, y_train,
        cv_folds=5
    )
    
    print("=== 훈련 결과 ===")
    for model_name, result in training_results.items():
        print(f"{model_name}:")
        print(f"  최고 점수: {result.best_score:.4f}")
        print(f"  최적 파라미터: {result.best_params}")
        print(f"  CV 점수: {result.cv_scores}")
        print()
    
    # 검증 데이터로 평가
    validation_results = trainer.evaluate_on_validation(X_val, y_val)
    
    print("=== 검증 결과 ===")
    for model_name, metrics in validation_results.items():
        print(f"{model_name}: {metrics}")
    
    # 모델 저장
    saved_paths = trainer.save_models(output_dir="models")
    print(f"모델 저장 완료: {saved_paths}")
    
    return training_results, validation_results

def evaluate_models():
    """모델 평가 및 비교"""
    
    # 가상의 테스트 데이터
    np.random.seed(123)
    X_test = np.random.randn(30, 30)
    y_test = np.random.choice([0, 1, 2], size=30)
    
    # 평가기 초기화
    evaluator = ModelEvaluator(config=DEFAULT_CONFIG)
    
    # 저장된 모델 로드 및 평가
    model_dir = "models"
    evaluation_results = evaluator.evaluate_saved_models(
        model_dir=model_dir,
        X_test=X_test,
        y_test=y_test
    )
    
    print("=== 테스트 평가 결과 ===")
    for model_name, result in evaluation_results.items():
        metrics = result.classification_metrics
        print(f"{model_name}:")
        print(f"  정확도: {metrics.accuracy:.4f}")
        print(f"  F1-점수: {metrics.macro_f1:.4f}")
        print(f"  ROC AUC: {metrics.roc_auc:.4f}")
        print()
    
    # 모델 비교
    comparison = evaluator.compare_models(evaluation_results)
    print("=== 모델 비교 ===")
    print(f"최고 성능 모델: {comparison.best_model}")
    print(f"통계적 유의성: p-value = {comparison.p_value:.6f}")
    
    # 평가 보고서 저장
    evaluator.save_evaluation_report(
        evaluation_results, 
        output_path="results/evaluation_report.json"
    )
    
    return evaluation_results

if __name__ == "__main__":
    # 모델 훈련
    training_results, validation_results = train_models()
    
    # 모델 평가
    evaluation_results = evaluate_models()
```

---

## 모델 변환 및 배포

### 1. Core ML 변환

```python
"""
Core ML 모델 변환 예제
"""

from src.ml.model_converter import ModelConverter
from config import DEFAULT_CONFIG

def convert_to_coreml():
    """훈련된 모델을 Core ML로 변환"""
    
    converter = ModelConverter(config=DEFAULT_CONFIG)
    
    # Pickle 모델 경로
    model_paths = {
        'svm': 'models/svm_model.pkl',
        'random_forest': 'models/random_forest_model.pkl'
    }
    
    conversion_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"=== {model_name.upper()} 모델 변환 ===")
        
        try:
            # Core ML 변환
            result = converter.convert_to_coreml(
                model_path=model_path,
                output_path=f"models/{model_name}_model.mlmodel",
                model_name=f"WatermelonClassifier_{model_name.upper()}",
                model_description=f"수박 숙성도 분류 모델 ({model_name})"
            )
            
            if result.success:
                print(f"✅ 변환 성공!")
                print(f"  출력 경로: {result.output_path}")
                print(f"  모델 크기: {result.model_size_mb:.2f} MB")
                print(f"  변환 시간: {result.conversion_time:.3f}초")
                
                # 변환 검증
                if result.validation_passed:
                    print(f"  검증 통과: 평균 차이 {result.validation_score:.6f}")
                else:
                    print(f"  ⚠️ 검증 실패: 예측 결과 불일치")
                
                conversion_results[model_name] = result
                
            else:
                print(f"❌ 변환 실패: {result.error_message}")
                
        except Exception as e:
            print(f"❌ 변환 중 오류: {e}")
        
        print()
    
    # 변환 요약 저장
    if conversion_results:
        summary = converter.get_conversion_summary(list(conversion_results.values()))
        converter.save_conversion_summary(summary, "results/conversion_summary.json")
        print(f"변환 요약 저장: results/conversion_summary.json")
    
    return conversion_results

def use_coreml_model():
    """변환된 Core ML 모델 사용 예제"""
    
    try:
        import coremltools as ct
        import numpy as np
        
        # Core ML 모델 로드
        model_path = "models/svm_model.mlmodel"
        model = ct.models.MLModel(model_path)
        
        print(f"모델 로드 완료: {model_path}")
        print(f"모델 설명: {model.short_description}")
        
        # 입력 사양 확인
        input_spec = model.get_spec().description.input[0]
        print(f"입력 형태: {input_spec.name} - {input_spec.type}")
        
        # 예측 수행
        sample_features = np.random.randn(1, 30).astype(np.float32)
        
        # Core ML 입력 형식으로 변환
        input_dict = {"features": sample_features}
        
        # 예측
        prediction = model.predict(input_dict)
        
        print(f"입력 특징: {sample_features[0][:5]}...")
        print(f"예측 결과: {prediction}")
        
        return prediction
        
    except ImportError:
        print("coremltools가 설치되지 않았습니다: pip install coremltools")
        return None
    except Exception as e:
        print(f"Core ML 모델 사용 중 오류: {e}")
        return None

if __name__ == "__main__":
    # Core ML 변환
    conversion_results = convert_to_coreml()
    
    # 변환된 모델 사용
    if conversion_results:
        prediction = use_coreml_model()
```

---

## 유틸리티 함수 활용

### 1. 파일 유틸리티 사용

```python
"""
파일 유틸리티 사용 예제
"""

from src.utils.file_utils import FileUtils, JsonUtils, PickleUtils, AudioFileUtils
from pathlib import Path

def file_operations_example():
    """파일 작업 예제"""
    
    # 디렉토리 생성
    output_dir = FileUtils.ensure_directory("examples/output")
    print(f"디렉토리 생성: {output_dir}")
    
    # 파일 검색
    audio_files = FileUtils.find_files(
        directory="data/raw",
        pattern="*.wav",
        recursive=True,
        file_types=['.wav', '.mp3']
    )
    print(f"발견된 오디오 파일: {len(audio_files)}개")
    
    # 파일 정보 조회
    if audio_files:
        file_info = FileUtils.get_file_info(audio_files[0])
        print(f"파일 정보: {file_info}")
        
        # 파일 해시 계산
        file_hash = FileUtils.get_file_hash(audio_files[0])
        print(f"파일 해시: {file_hash}")

def json_operations_example():
    """JSON 작업 예제"""
    
    # 데이터 준비
    config_data = {
        "model_params": {
            "svm": {"C": 1.0, "gamma": "scale"},
            "rf": {"n_estimators": 100, "max_depth": 10}
        },
        "feature_config": {
            "n_mfcc": 13,
            "n_mels": 128,
            "hop_length": 512
        },
        "training_config": {
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42
        }
    }
    
    # JSON 저장
    json_path = JsonUtils.save_json(
        config_data, 
        "examples/output/model_config.json"
    )
    print(f"JSON 저장: {json_path}")
    
    # JSON 로드
    loaded_data = JsonUtils.load_json(json_path)
    print(f"로드된 데이터 키: {list(loaded_data.keys())}")
    
    # JSON 업데이트
    updates = {"last_updated": "2024-01-15", "version": "1.0"}
    updated_data = JsonUtils.update_json(json_path, updates)
    print(f"업데이트된 항목: {updates}")

def audio_validation_example():
    """오디오 파일 검증 예제"""
    
    # 오디오 파일 검색
    audio_files = list(Path("data/raw").rglob("*.wav"))[:5]  # 처음 5개만
    
    print("=== 개별 파일 검증 ===")
    for audio_file in audio_files:
        # 오디오 정보 조회
        audio_info = AudioFileUtils.get_audio_info(audio_file)
        print(f"\n{audio_file.name}:")
        if 'error' not in audio_info:
            print(f"  지속시간: {audio_info.get('duration_seconds', 'N/A')}초")
            print(f"  샘플 레이트: {audio_info.get('sample_rate', 'N/A')} Hz")
            print(f"  크기: {audio_info['size_mb']:.2f} MB")
        
        # 파일 검증
        validation = AudioFileUtils.validate_audio_file(audio_file)
        status = "✅ 유효" if validation['is_valid'] else "❌ 무효"
        print(f"  검증 상태: {status}")
        
        if validation['warnings']:
            print(f"  경고: {', '.join(validation['warnings'])}")
        if validation['errors']:
            print(f"  오류: {', '.join(validation['errors'])}")
    
    # 배치 검증
    print("\n=== 배치 검증 ===")
    batch_results = AudioFileUtils.batch_validate_audio_files("data/raw")
    print(f"전체 파일: {batch_results['total_files']}개")
    print(f"유효한 파일: {batch_results['valid_files']}개")
    print(f"무효한 파일: {batch_results['invalid_files']}개")
    print(f"경고가 있는 파일: {batch_results['files_with_warnings']}개")
    print(f"총 재생 시간: {batch_results['summary']['total_duration']:.1f}초")

if __name__ == "__main__":
    # 파일 작업 예제
    file_operations_example()
    
    # JSON 작업 예제  
    json_operations_example()
    
    # 오디오 검증 예제
    audio_validation_example()
```

### 2. 배열 및 메모리 유틸리티

```python
"""
배열 및 메모리 유틸리티 예제
"""

import numpy as np
from src.utils.file_utils import ArrayUtils, MemoryUtils

def array_processing_example():
    """배열 처리 예제"""
    
    # 가상 데이터 생성
    np.random.seed(42)
    X = np.random.randn(1000, 30) * 10 + 5  # 평균 5, 표준편차 10
    y = np.random.choice([0, 1, 2], size=1000)
    
    print("=== 데이터 정규화 ===")
    
    # 표준 정규화
    X_std, std_params = ArrayUtils.normalize_features(X, method='standard')
    print(f"표준 정규화 후 평균: {np.mean(X_std, axis=0)[:3]}")
    print(f"표준 정규화 후 표준편차: {np.std(X_std, axis=0)[:3]}")
    
    # MinMax 정규화
    X_minmax, minmax_params = ArrayUtils.normalize_features(X, method='minmax')
    print(f"MinMax 정규화 후 최소값: {np.min(X_minmax, axis=0)[:3]}")
    print(f"MinMax 정규화 후 최대값: {np.max(X_minmax, axis=0)[:3]}")
    
    # 정규화 파라미터 적용
    X_new = np.random.randn(10, 30) * 10 + 5
    X_new_normalized = ArrayUtils.apply_normalization(X_new, std_params)
    print(f"새 데이터 정규화 완료: {X_new_normalized.shape}")
    
    print("\\n=== 데이터 분할 ===")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = ArrayUtils.split_array(
        X_std, y, test_size=0.2, stratify=True
    )
    print(f"훈련 세트: {X_train.shape}, 테스트 세트: {X_test.shape}")
    print(f"훈련 클래스 분포: {np.bincount(y_train)}")
    print(f"테스트 클래스 분포: {np.bincount(y_test)}")
    
    print("\\n=== 이상치 제거 ===")
    
    # 이상치 제거
    X_clean, outlier_mask = ArrayUtils.remove_outliers(X_train, method='iqr')
    print(f"원본 데이터: {X_train.shape}")
    print(f"정리된 데이터: {X_clean.shape}")
    print(f"제거된 이상치: {np.sum(outlier_mask)}개")
    
    print("\\n=== 특징 중요도 ===")
    
    # 특징 중요도 계산
    feature_names = [f'Feature_{i+1}' for i in range(30)]
    importance = ArrayUtils.compute_feature_importance(X_clean, y_train[~outlier_mask], feature_names)
    
    print("상위 10개 중요한 특징:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1}. {feature}: {score:.4f}")

def memory_management_example():
    """메모리 관리 예제"""
    
    print("=== 메모리 사용량 모니터링 ===")
    
    # 현재 메모리 사용량
    memory_info = MemoryUtils.get_memory_usage()
    if 'error' not in memory_info:
        print(f"현재 메모리 사용량: {memory_info['rss_mb']:.2f} MB")
        print(f"메모리 사용률: {memory_info['percent']:.1f}%")
    
    # 큰 배열 생성 (메모리 사용량 증가)
    print("\\n대용량 배열 생성 중...")
    large_array = np.random.randn(10000, 100)  # ~80MB
    
    estimated_memory = MemoryUtils.estimate_array_memory(large_array.shape, large_array.dtype)
    print(f"예상 메모리 사용량: {estimated_memory:.2f} MB")
    
    # 메모리 사용량 재측정
    memory_after = MemoryUtils.get_memory_usage()
    if 'error' not in memory_after and 'error' not in memory_info:
        memory_diff = memory_after['rss_mb'] - memory_info['rss_mb']
        print(f"실제 메모리 증가: {memory_diff:.2f} MB")
    
    print("\\n=== 메모리 효율적 처리 ===")
    
    # 배열 청킹
    chunks = MemoryUtils.chunk_array(large_array, max_memory_mb=20)
    print(f"원본 배열: {large_array.shape}")
    print(f"청크 수: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chunk_memory = MemoryUtils.estimate_array_memory(chunk.shape, chunk.dtype)
        print(f"  청크 {i+1}: {chunk.shape}, 메모리: {chunk_memory:.2f} MB")
    
    # 배치 처리
    def simple_processing(x):
        return np.mean(x, axis=1)
    
    # 메모리 효율적 연산
    result = MemoryUtils.memory_efficient_operation(
        large_array, 
        simple_processing,
        max_memory_mb=30
    )
    print(f"\\n처리 결과 크기: {result.shape}")
    
    # 메모리 정리
    del large_array
    cleanup_result = MemoryUtils.clear_memory()
    print(f"가비지 컬렉션: {cleanup_result['collected_objects']}개 객체 정리")

@MemoryUtils.monitor_memory_usage
def memory_monitored_function():
    """메모리 모니터링 데코레이터 예제"""
    
    # 메모리를 사용하는 작업
    data = np.random.randn(5000, 50)
    result = np.dot(data, data.T)
    
    return result.shape

if __name__ == "__main__":
    # 배열 처리 예제
    array_processing_example()
    
    # 메모리 관리 예제
    memory_management_example()
    
    # 메모리 모니터링 데코레이터 예제
    print("\\n=== 메모리 모니터링 데코레이터 ===")
    result_shape = memory_monitored_function()
    print(f"함수 결과: {result_shape}")
```

---

## 성능 모니터링

### 1. 파이프라인 성능 분석

```python
"""
성능 모니터링 예제
"""

from src.utils.performance_monitor import PerformanceMonitor
from src.utils.data_integrity import DataIntegrityChecker
import numpy as np
import time

def performance_monitoring_example():
    """성능 모니터링 사용 예제"""
    
    # 성능 모니터 초기화
    monitor = PerformanceMonitor(enable_monitoring=True)
    
    print("=== 단계별 성능 모니터링 ===")
    
    # 1단계: 데이터 로딩 시뮬레이션
    monitor.start_step_monitoring("data_loading")
    
    # 데이터 로딩 작업 시뮬레이션
    time.sleep(0.5)
    large_data = np.random.randn(5000, 100)
    
    loading_metrics = monitor.end_step_monitoring(
        "data_loading", 
        {"data_shape": large_data.shape, "data_type": str(large_data.dtype)}
    )
    
    # 2단계: 특징 추출 시뮬레이션
    monitor.start_step_monitoring("feature_extraction")
    
    # 특징 추출 작업 시뮬레이션
    features = np.mean(large_data, axis=1, keepdims=True)
    features = np.concatenate([features, np.std(large_data, axis=1, keepdims=True)], axis=1)
    time.sleep(0.3)
    
    extraction_metrics = monitor.end_step_monitoring(
        "feature_extraction",
        {"feature_shape": features.shape, "extraction_method": "mean_std"}
    )
    
    # 3단계: 모델 훈련 시뮬레이션
    monitor.start_step_monitoring("model_training")
    
    # 훈련 작업 시뮬레이션
    time.sleep(1.0)
    model_params = {"n_estimators": 100, "max_depth": 10}
    
    training_metrics = monitor.end_step_monitoring(
        "model_training",
        {"model_params": model_params, "training_samples": len(features)}
    )
    
    print("\\n=== 성능 요약 ===")
    
    # 성능 요약 가져오기
    summary = monitor.get_performance_summary()
    
    if 'overall_statistics' in summary:
        stats = summary['overall_statistics']
        print(f"전체 실행 시간: {stats['total_execution_time']:.3f}초")
        print(f"전체 메모리 사용: {stats['total_memory_used']:+.2f} MB")
        print(f"측정된 단계: {stats['unique_steps']}개")
        
        # 가장 느린/빠른 단계
        insights = summary['performance_insights']
        print(f"\\n가장 느린 단계: {insights['slowest_step']['name']} ({insights['slowest_step']['avg_time']:.3f}초)")
        print(f"가장 빠른 단계: {insights['fastest_step']['name']} ({insights['fastest_step']['avg_time']:.3f}초)")
        print(f"메모리 집약적 단계: {insights['memory_intensive_step']['name']} ({insights['memory_intensive_step']['avg_memory']:+.2f} MB)")
    
    # 성능 비교
    comparison = monitor.compare_performance("data_loading", "model_training")
    if 'error' not in comparison:
        print(f"\\n=== 단계별 비교 (데이터 로딩 vs 모델 훈련) ===")
        exec_improvement = comparison['execution_time']['improvement']
        memory_improvement = comparison['memory_usage']['improvement']
        print(f"실행 시간 차이: {exec_improvement}")
        print(f"메모리 사용 차이: {memory_improvement}")
    
    # 성능 보고서 저장
    monitor.save_performance_report("results/performance_report.json")
    print("\\n성능 보고서 저장: results/performance_report.json")
    
    return monitor

def data_integrity_monitoring_example():
    """데이터 무결성 모니터링 예제"""
    
    # 무결성 검사기 초기화
    from config import DEFAULT_CONFIG
    checker = DataIntegrityChecker(config=DEFAULT_CONFIG)
    
    print("=== 데이터 무결성 검사 ===")
    
    # 가상 데이터 생성
    np.random.seed(42)
    X = np.random.randn(100, 30)
    y = np.random.choice([0, 1, 2], size=100)
    
    # 특징 데이터 무결성 검사
    feature_report = checker.check_audio_features(X, y, "test_features")
    
    print(f"특징 검사 결과:")
    print(f"  통과: {feature_report.passed}")
    print(f"  성공률: {feature_report.success_rate:.1%}")
    print(f"  경고: {len(feature_report.warnings)}개")
    print(f"  오류: {len(feature_report.errors)}개")
    
    # 모델 출력 무결성 검사 (가상 예측 결과)
    predictions = np.random.choice([0, 1, 2], size=50)
    probabilities = np.random.dirichlet([1, 1, 1], size=50)  # 합이 1인 확률
    true_labels = np.random.choice([0, 1, 2], size=50)
    
    model_report = checker.check_model_outputs(
        predictions, probabilities, true_labels, "test_model_output"
    )
    
    print(f"\\n모델 출력 검사 결과:")
    print(f"  통과: {model_report.passed}")
    print(f"  성공률: {model_report.success_rate:.1%}")
    if 'accuracy_info' in model_report.details:
        accuracy = model_report.details['accuracy_info']['accuracy']
        print(f"  예측 정확도: {accuracy:.3f}")
    
    # 전체 요약
    summary = checker.get_summary_report()
    print(f"\\n=== 전체 무결성 요약 ===")
    if 'overall_statistics' in summary:
        stats = summary['overall_statistics']
        print(f"총 검사 단계: {stats['total_checks_run']}개")
        print(f"전체 성공률: {stats['overall_success_rate']:.1%}")
        print(f"전체 상태: {summary['overall_status']}")
    
    return checker

def benchmark_operation_example():
    """특정 연산 벤치마킹 예제"""
    
    monitor = PerformanceMonitor(enable_monitoring=True)
    
    def matrix_multiplication(size):
        """행렬 곱셈 연산"""
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        return np.dot(A, B)
    
    print("=== 연산 벤치마킹 ===")
    
    # 다양한 크기로 벤치마킹
    sizes = [100, 200, 500]
    
    for size in sizes:
        result, metrics = monitor.benchmark_operation(
            matrix_multiplication, 
            f"matrix_mult_{size}x{size}",
            size
        )
        
        print(f"행렬 크기 {size}x{size}:")
        print(f"  실행 시간: {metrics.execution_time:.3f}초")
        print(f"  메모리 사용: {metrics.memory_used:+.2f} MB")
        print(f"  결과 크기: {result.shape}")
        print()
    
    # 벤치마킹 결과 비교
    comparison_100_500 = monitor.compare_performance(
        "matrix_mult_100x100", 
        "matrix_mult_500x500"
    )
    
    if 'error' not in comparison_100_500:
        print("100x100 vs 500x500 행렬 곱셈 비교:")
        print(f"  실행 시간: {comparison_100_500['execution_time']['improvement']}")
        print(f"  메모리 사용: {comparison_100_500['memory_usage']['improvement']}")
    
    return monitor

if __name__ == "__main__":
    # 성능 모니터링 예제
    monitor = performance_monitoring_example()
    
    # 데이터 무결성 모니터링 예제
    checker = data_integrity_monitoring_example()
    
    # 벤치마킹 예제
    benchmark_monitor = benchmark_operation_example()
```

---

## 결론

이 예제들은 수박 소리 분류 시스템의 다양한 구성요소를 효과적으로 활용하는 방법을 보여줍니다:

1. **파이프라인 실행**: 전체 시스템을 명령줄이나 프로그래밍 방식으로 실행
2. **구성요소별 사용**: 특징 추출, 데이터 증강, 모델 훈련 등을 개별적으로 활용
3. **유틸리티 활용**: 파일 처리, 배열 조작, 시각화, 메모리 관리 등의 보조 기능
4. **품질 관리**: 성능 모니터링과 데이터 무결성 검사를 통한 시스템 신뢰성 확보

각 예제는 실제 프로젝트에서 바로 사용할 수 있도록 구성되었으며, 오류 처리와 로깅도 포함되어 있습니다.