# 수박 소리 분류 시스템 최종 배포 가이드

본 문서는 수박 소리 분류 시스템의 최종 배포와 운영을 위한 종합 가이드입니다.

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [배포 전 체크리스트](#배포-전-체크리스트)
3. [모델 배포 방법](#모델-배포-방법)
4. [성능 최적화](#성능-최적화)
5. [모니터링 및 유지보수](#모니터링-및-유지보수)
6. [문제 해결](#문제-해결)

## 🎯 시스템 개요

### 핵심 기능
- **오디오 특징 추출**: 30차원 특징 벡터 (MFCC, Mel Spectrogram, Spectral Features, Chroma)
- **데이터 증강**: SNR 제어 기반 노이즈 혼합
- **머신러닝 모델**: SVM, Random Forest 분류기
- **다중 배포 형식**: Python Pickle, Core ML
- **성능 최적화**: 하드웨어별 자동 최적화, 병렬 처리, 메모리 효율성

### 지원 플랫폼
- **Python**: pickle 모델을 사용한 서버/데스크톱 배포
- **iOS/macOS**: Core ML 모델을 사용한 모바일/데스크톱 앱
- **웹**: Flask/FastAPI를 통한 REST API 서비스

## ✅ 배포 전 체크리스트

### 1. 시스템 요구사항 검증
```bash
# 요구사항 자동 검증
cd /Users/parksingyu/PycharmProjects/wm_mlt_models
python validation/requirements_verification.py

# 최소 요구사항
# - Python 3.7+
# - CPU: 2코어 이상 (권장: 4코어)
# - RAM: 2GB 이상 (권장: 8GB)
# - 디스크: 1GB 여유 공간 (권장: 5GB)
```

### 2. 전체 시스템 검증
```bash
# 포괄적 파이프라인 검증
python validation/comprehensive_pipeline_validation.py

# 빠른 검증 (시간 절약)
python validation/comprehensive_pipeline_validation.py --skip-slow
```

### 3. 배포 준비 상태 확인
```bash
# 모델 배포 준비 상태 검증
python validation/deployment_readiness_checker.py

# 결과 해석:
# - READY: 즉시 배포 가능
# - PARTIALLY_READY: 일부 개선 후 배포 가능  
# - NOT_READY: 문제 해결 필요
```

### 4. 리소스 정리
```bash
# 배포 전 리소스 정리
python src/utils/resource_cleanup.py

# 시뮬레이션 (실제 삭제 X)
python src/utils/resource_cleanup.py --dry-run
```

## 🚀 모델 배포 방법

### Python 서버 배포

#### 1. 기본 배포
```python
# 단순 모델 로딩 및 예측
import pickle
import numpy as np
from src.audio.feature_extraction import extract_features
from config import DEFAULT_CONFIG

# 모델 로딩
with open('results/trained_models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 예측 함수
def predict_watermelon_class(audio_file_path):
    """수박 소리 분류 예측"""
    features = extract_features(audio_file_path, DEFAULT_CONFIG)
    if features is None:
        return None
    
    feature_vector = features.to_array().reshape(1, -1)
    prediction = model.predict(feature_vector)[0]
    confidence = model.predict_proba(feature_vector).max()
    
    class_names = ['watermelon_A', 'watermelon_B', 'watermelon_C']
    return {
        'class': class_names[prediction],
        'confidence': float(confidence)
    }
```

#### 2. 최적화된 배포
```python
# 최적화된 배포를 위한 통합 시스템 사용
from src.optimization.integrated_optimizer import IntegratedOptimizer

# 자동 최적화기 생성
optimizer = IntegratedOptimizer()

# 배치 예측 (대용량 처리)
def predict_batch_optimized(audio_files):
    """최적화된 배치 예측"""
    result = optimizer.process_dataset_optimized(
        audio_files=audio_files,
        extract_features=True,
        perform_augmentation=False
    )
    return result
```

#### 3. Flask 웹 서비스
```python
# app.py
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

# 모델 로딩 (앱 시작시 한 번만)
with open('results/trained_models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """오디오 파일 업로드 및 예측"""
    if 'audio' not in request.files:
        return jsonify({'error': '오디오 파일이 필요합니다'}), 400
    
    audio_file = request.files['audio']
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        audio_file.save(tmp_file.name)
        
        try:
            result = predict_watermelon_class(tmp_file.name)
            return jsonify(result)
        finally:
            os.unlink(tmp_file.name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Core ML 모바일 배포

#### 1. iOS 배포 (Swift)
```swift
// WatermelonClassifier.swift
import CoreML
import AVFoundation

class WatermelonClassifier {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "watermelon_classifier", withExtension: "mlmodel") else {
            print("모델 파일을 찾을 수 없습니다")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
        } catch {
            print("모델 로딩 실패: \(error)")
        }
    }
    
    func predict(audioFeatures: [Double]) -> (className: String, confidence: Double)? {
        guard let model = model else { return nil }
        
        // 특징 벡터를 MLMultiArray로 변환
        guard let featureArray = try? MLMultiArray(shape: [30], dataType: .double) else {
            return nil
        }
        
        for (index, value) in audioFeatures.enumerated() {
            featureArray[index] = NSNumber(value: value)
        }
        
        // 예측 수행
        do {
            let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": featureArray]))
            
            if let output = prediction.featureValue(for: "classLabel")?.stringValue,
               let confidence = prediction.featureValue(for: "classProbability")?.dictionaryValue {
                
                let maxConfidence = confidence.values.compactMap { $0.doubleValue }.max() ?? 0.0
                return (className: output, confidence: maxConfidence)
            }
        } catch {
            print("예측 실패: \(error)")
        }
        
        return nil
    }
}
```

#### 2. Android 배포 (Java/Kotlin)
```kotlin
// WatermelonClassifier.kt (TensorFlow Lite 변환 필요시)
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WatermelonClassifier(private val modelPath: String) {
    private var interpreter: Interpreter? = null
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            val model = loadModelFile(modelPath)
            interpreter = Interpreter(model)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    fun predict(features: FloatArray): Pair<String, Float>? {
        val interpreter = this.interpreter ?: return null
        
        // 입력 준비
        val inputBuffer = ByteBuffer.allocateDirect(4 * features.size)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        features.forEach { inputBuffer.putFloat(it) }
        
        // 출력 준비
        val outputBuffer = ByteBuffer.allocateDirect(4 * 3) // 3 classes
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // 예측
        interpreter.run(inputBuffer, outputBuffer)
        
        // 결과 파싱
        outputBuffer.rewind()
        val probabilities = FloatArray(3)
        outputBuffer.asFloatBuffer().get(probabilities)
        
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val classNames = arrayOf("watermelon_A", "watermelon_B", "watermelon_C")
        
        return Pair(classNames[maxIndex], probabilities[maxIndex])
    }
}
```

## ⚡ 성능 최적화

### 1. 하드웨어별 자동 최적화
```python
# 시스템 하드웨어에 맞는 자동 최적화
from src.config.hardware_config import get_hardware_config

# 현재 하드웨어 설정 확인
config_manager = get_hardware_config()
config_manager.print_system_info()

# 프리셋 변경 (필요시)
config_manager.set_preset('high_performance')  # 또는 'balanced', 'low_memory'
```

### 2. 배치 처리 최적화
```python
# 대용량 파일 처리
from src.audio.batch_processor import extract_features_batch, BatchProcessingConfig

# 최적화된 배치 설정
config = BatchProcessingConfig(
    max_workers=4,
    chunk_size=32,
    use_multiprocessing=True,
    cache_features=True
)

# 배치 특징 추출
audio_files = ['file1.wav', 'file2.wav', ...]
result = extract_features_batch(audio_files, config)

print(f"처리된 파일: {result.success_count}/{len(audio_files)}")
print(f"총 시간: {result.total_time:.2f}초")
print(f"평균 속도: {result.success_count/result.total_time:.1f} files/sec")
```

### 3. 메모리 효율적 처리
```python
# 대용량 데이터셋 스트리밍 처리
from src.data.large_dataset_processor import process_large_dataset_memory_efficient

result = process_large_dataset_memory_efficient(
    audio_files=large_file_list,
    memory_limit_gb=2.0,  # 메모리 제한
    include_features=True,
    progress_callback=lambda p: print(f"진행률: {p:.1%}")
)
```

### 4. 병렬 데이터 증강
```python
# 병렬 데이터 증강
from src.data.parallel_augmentor import augment_directory_parallel, ParallelAugmentationConfig

config = ParallelAugmentationConfig(
    max_workers=4,
    chunk_size=16,
    snr_levels=[0, 5, 10],
    use_multiprocessing=True
)

result = augment_directory_parallel(
    audio_dir='data/raw/train/watermelon_A',
    noise_dir='data/noise',
    output_dir='data/augmented',
    config=config
)
```

## 📊 모니터링 및 유지보수

### 1. 성능 벤치마크
```python
# 시스템 성능 벤치마크
from src.optimization.integrated_optimizer import IntegratedOptimizer

optimizer = IntegratedOptimizer()
benchmark = optimizer.benchmark_system()

print(f"특징 추출 속도: {benchmark['feature_extraction']['files_per_second']:.1f} files/sec")
print(f"메모리 효율성: {benchmark['feature_extraction']['memory_efficiency']:.1%}")
```

### 2. 모델 성능 모니터링
```python
# 예측 결과 로깅 및 모니터링
import logging
from datetime import datetime

def predict_with_monitoring(audio_file):
    """모니터링이 포함된 예측"""
    start_time = time.time()
    
    try:
        result = predict_watermelon_class(audio_file)
        processing_time = time.time() - start_time
        
        # 성능 로깅
        logging.info(f"예측 성공: {audio_file}, 시간: {processing_time:.3f}초, "
                    f"결과: {result['class']}, 신뢰도: {result['confidence']:.3f}")
        
        return result
        
    except Exception as e:
        logging.error(f"예측 실패: {audio_file}, 오류: {str(e)}")
        return None
```

### 3. 자동 리소스 정리
```python
# 주기적 리소스 정리 (크론잡 등에서 실행)
from src.utils.resource_cleanup import quick_cleanup

# 24시간 이전 임시 파일 정리
report = quick_cleanup(preserve_hours=24, dry_run=False)
print(f"정리된 공간: {report.disk_space_freed_mb:.1f}MB")
```

### 4. 모델 업데이트 및 A/B 테스트
```python
# 모델 버전 관리
class ModelVersionManager:
    def __init__(self):
        self.models = {}
        self.current_version = 'v1.0'
    
    def load_model_version(self, version: str, model_path: str):
        """특정 버전 모델 로딩"""
        with open(model_path, 'rb') as f:
            self.models[version] = pickle.load(f)
    
    def predict_with_version(self, audio_file: str, version: str = None):
        """지정된 버전으로 예측"""
        version = version or self.current_version
        model = self.models.get(version)
        
        if not model:
            raise ValueError(f"모델 버전 {version}을 찾을 수 없습니다")
        
        # 예측 로직...
        return self._predict_with_model(model, audio_file)
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 메모리 부족 오류
```bash
# 해결 방법
# 1. 메모리 효율 모드 사용
python main.py --memory-limit 2.0

# 2. 스트리밍 처리 활성화
# src/optimization/integrated_optimizer.py에서 enable_streaming=True

# 3. 배치 크기 줄이기
# BatchProcessingConfig에서 chunk_size 감소
```

#### 2. 성능 저하
```python
# 성능 진단 및 최적화
from src.config.hardware_config import get_hardware_config

config = get_hardware_config()
config.print_system_info()

# 하드웨어에 맞는 프리셋 자동 적용
config.auto_configure()
```

#### 3. 모델 예측 오류
```python
# 모델 무결성 검증
from validation.deployment_readiness_checker import DeploymentReadinessChecker

checker = DeploymentReadinessChecker()
report = checker.run_comprehensive_deployment_check()

if report.overall_status != "READY":
    print("모델 문제 발견:")
    for issue in report.critical_issues:
        print(f"  - {issue}")
```

#### 4. Core ML 변환 실패
```bash
# Core ML 도구 설치 및 업데이트
pip install --upgrade coremltools

# Python 환경에서 변환 테스트
python -c "
from src.ml.model_converter import ModelConverter
converter = ModelConverter()
# 변환 테스트 코드...
"
```

### 로그 분석
```bash
# 로그 파일 위치
ls logs/

# 최근 오류 확인
tail -f logs/main.log | grep ERROR

# 성능 로그 분석
grep "처리 시간" logs/main.log | tail -20
```

### 디버깅 모드 실행
```bash
# 상세 디버깅 정보와 함께 실행
python main.py --debug --verbose

# 특정 모듈 디버깅
export PYTHONPATH=.
python -m src.audio.feature_extraction --debug
```

## 📞 지원 및 연락처

### 개발팀 연락처
- **기술 지원**: tech-support@company.com
- **버그 리포트**: GitHub Issues
- **성능 최적화 문의**: performance@company.com

### 참고 문서
- [API 레퍼런스](API_REFERENCE.md)
- [모델 사용 예제](MODEL_USAGE_EXAMPLES.md)
- [사용법 예제](USAGE_EXAMPLES.md)
- [Core ML 사용법](COREML_USAGE.md)

---

**최종 업데이트**: 2024년 12월 
**문서 버전**: 1.0
**시스템 버전**: 최종 완성 버전