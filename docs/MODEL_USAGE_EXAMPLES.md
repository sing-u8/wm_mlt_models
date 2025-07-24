# 모델 사용 예제 가이드

이 문서는 수박 소리 분류 시스템에서 생성된 Pickle 및 Core ML 모델을 다양한 환경에서 사용하는 방법을 설명합니다.

## 목차

1. [Pickle 모델 사용](#pickle-모델-사용)
2. [Core ML 모델 사용](#core-ml-모델-사용)
3. [배치 예측](#배치-예측)
4. [실시간 스트리밍](#실시간-스트리밍)
5. [성능 최적화](#성능-최적화)
6. [모바일 앱 통합](#모바일-앱-통합)

---

## Pickle 모델 사용

### 1. 기본 사용법

```python
"""
Pickle 모델로 수박 숙성도 예측하기
"""

import pickle
import numpy as np
from src.audio.feature_extraction import AudioFeatureExtractor

# 모델 로드
def load_watermelon_model(model_path="data/models/pickle/svm_model.pkl"):
    """훈련된 수박 분류 모델을 로드합니다."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# 특징 추출기 초기화
extractor = AudioFeatureExtractor()

# 모델 로드
model = load_watermelon_model()

def predict_watermelon_ripeness(audio_path):
    """
    수박 소리 파일로부터 숙성도를 예측합니다.
    
    Args:
        audio_path (str): 수박 소리 파일 경로
        
    Returns:
        dict: 예측 결과와 신뢰도
    """
    try:
        # 특징 추출
        features = extractor.extract_features(audio_path)
        features = features.reshape(1, -1)
        
        # 예측
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # 클래스별 확률
        classes = ['watermelon_A', 'watermelon_B', 'watermelon_C']
        class_probabilities = {
            cls: prob for cls, prob in zip(classes, probabilities)
        }
        
        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'probabilities': class_probabilities,
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }

# 사용 예제
if __name__ == "__main__":
    # 단일 파일 예측
    audio_file = "data/raw/test/watermelon_A/sample_001.wav"
    result = predict_watermelon_ripeness(audio_file)
    
    if result['status'] == 'success':
        print(f"예측 결과: {result['prediction']}")
        print(f"신뢰도: {result['confidence']:.4f}")
        print("클래스별 확률:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    else:
        print(f"예측 실패: {result['error_message']}")
```

### 2. 모델 성능 검증

```python
"""
모델 성능을 검증하는 예제
"""

import glob
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def validate_model_performance(model_path, test_data_dir):
    """
    테스트 데이터로 모델 성능을 검증합니다.
    
    Args:
        model_path (str): 모델 파일 경로
        test_data_dir (str): 테스트 데이터 디렉토리
        
    Returns:
        dict: 성능 메트릭
    """
    model = load_watermelon_model(model_path)
    extractor = AudioFeatureExtractor()
    
    predictions = []
    true_labels = []
    
    # 각 클래스별로 테스트
    for class_name in ['watermelon_A', 'watermelon_B', 'watermelon_C']:
        class_dir = f"{test_data_dir}/{class_name}"
        audio_files = glob.glob(f"{class_dir}/*.wav")
        
        for audio_file in audio_files:
            try:
                # 특징 추출 및 예측
                features = extractor.extract_features(audio_file)
                features = features.reshape(1, -1)
                prediction = model.predict(features)[0]
                
                predictions.append(prediction)
                true_labels.append(class_name)
                
            except Exception as e:
                print(f"예측 실패 {audio_file}: {e}")
    
    # 성능 계산
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'total_samples': len(predictions)
    }

# 모델 검증 실행
validation_result = validate_model_performance(
    "data/models/pickle/svm_model.pkl",
    "data/raw/test"
)

print(f"모델 정확도: {validation_result['accuracy']:.4f}")
print(f"테스트 샘플 수: {validation_result['total_samples']}")
```

### 3. 다중 모델 앙상블

```python
"""
여러 모델을 결합한 앙상블 예측
"""

class WatermelonEnsemble:
    """수박 분류를 위한 앙상블 모델"""
    
    def __init__(self, model_paths):
        self.models = {}
        self.extractor = AudioFeatureExtractor()
        
        # 모든 모델 로드
        for name, path in model_paths.items():
            self.models[name] = load_watermelon_model(path)
    
    def predict_ensemble(self, audio_path, method='voting'):
        """
        앙상블 예측을 수행합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
            method (str): 앙상블 방법 ('voting', 'average')
            
        Returns:
            dict: 앙상블 예측 결과
        """
        # 특징 추출
        features = self.extractor.extract_features(audio_path)
        features = features.reshape(1, -1)
        
        # 각 모델에서 예측
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            
            predictions[name] = pred
            probabilities[name] = prob
        
        if method == 'voting':
            # 다수결 투표
            from collections import Counter
            votes = list(predictions.values())
            ensemble_prediction = Counter(votes).most_common(1)[0][0]
            
        elif method == 'average':
            # 확률 평균
            classes = ['watermelon_A', 'watermelon_B', 'watermelon_C']
            avg_probs = np.mean([probabilities[name] for name in probabilities], axis=0)
            ensemble_prediction = classes[np.argmax(avg_probs)]
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'method': method
        }

# 앙상블 모델 사용
model_paths = {
    'svm': 'data/models/pickle/svm_model.pkl',
    'random_forest': 'data/models/pickle/random_forest_model.pkl'
}

ensemble = WatermelonEnsemble(model_paths)
result = ensemble.predict_ensemble("test_audio.wav", method='voting')
print(f"앙상블 예측: {result['ensemble_prediction']}")
```

---

## Core ML 모델 사용

### 1. Python에서 Core ML 모델 사용

```python
"""
Python 환경에서 Core ML 모델 사용하기
"""

import coremltools as ct
import numpy as np
from src.audio.feature_extraction import AudioFeatureExtractor

class CoreMLWatermelonClassifier:
    """Core ML 수박 분류기"""
    
    def __init__(self, model_path="data/models/coreml/svm_model.mlmodel"):
        """
        Core ML 모델을 로드합니다.
        
        Args:
            model_path (str): Core ML 모델 파일 경로
        """
        try:
            self.model = ct.models.MLModel(model_path)
            self.extractor = AudioFeatureExtractor()
            
            # 모델 정보 출력
            spec = self.model.get_spec()
            print(f"모델 로드 완료: {model_path}")
            print(f"모델 설명: {spec.description}")
            
        except Exception as e:
            raise RuntimeError(f"Core ML 모델 로드 실패: {e}")
    
    def predict(self, audio_path):
        """
        오디오 파일에 대해 예측을 수행합니다.
        
        Args:
            audio_path (str): 오디오 파일 경로
            
        Returns:
            dict: 예측 결과
        """
        try:
            # 특징 추출
            features = self.extractor.extract_features(audio_path)
            
            # Core ML 입력 형식으로 변환
            input_dict = {"audio_features": features.astype(np.float32)}
            
            # 예측
            prediction = self.model.predict(input_dict)
            
            return {
                'prediction': prediction['watermelon_class'],
                'probabilities': prediction['watermelon_class_proba'],
                'confidence': max(prediction['watermelon_class_proba'].values()),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def batch_predict(self, audio_paths):
        """
        여러 오디오 파일에 대해 배치 예측을 수행합니다.
        
        Args:
            audio_paths (list): 오디오 파일 경로 리스트
            
        Returns:
            list: 예측 결과 리스트
        """
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            result['file_path'] = audio_path
            results.append(result)
        return results

# Core ML 모델 사용 예제
if __name__ == "__main__":
    # 분류기 초기화
    classifier = CoreMLWatermelonClassifier()
    
    # 단일 예측
    result = classifier.predict("test_audio.wav")
    if result['status'] == 'success':
        print(f"예측: {result['prediction']}")
        print(f"신뢰도: {result['confidence']:.4f}")
        print("확률 분포:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    
    # 배치 예측
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    batch_results = classifier.batch_predict(audio_files)
    
    for result in batch_results:
        if result['status'] == 'success':
            print(f"{result['file_path']}: {result['prediction']} ({result['confidence']:.3f})")
```

### 2. Swift에서 Core ML 모델 사용 (iOS/macOS)

```swift
/*
 Swift iOS/macOS 앱에서 Core ML 모델 사용하기
 */

import CoreML
import Foundation

class WatermelonClassifier {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "svm_model", withExtension: "mlmodel") else {
            print("모델 파일을 찾을 수 없습니다.")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("Core ML 모델 로드 완료")
        } catch {
            print("Core ML 모델 로드 실패: \\(error)")
        }
    }
    
    func predict(features: [Float]) -> (prediction: String?, confidence: Double?) {
        guard let model = model else {
            print("모델이 로드되지 않았습니다.")
            return (nil, nil)
        }
        
        guard features.count == 30 else {
            print("특징 벡터는 30차원이어야 합니다.")
            return (nil, nil)
        }
        
        do {
            // 입력 데이터 준비
            let multiArray = try MLMultiArray(shape: [30], dataType: .float32)
            for (index, feature) in features.enumerated() {
                multiArray[index] = NSNumber(value: feature)
            }
            
            // 예측 수행
            let input = try MLDictionaryFeatureProvider(dictionary: ["audio_features": multiArray])
            let output = try model.prediction(from: input)
            
            // 결과 추출
            if let prediction = output.featureValue(for: "watermelon_class")?.stringValue,
               let probabilities = output.featureValue(for: "watermelon_class_proba")?.dictionaryValue {
                
                let confidence = probabilities.values.compactMap { $0.doubleValue }.max() ?? 0.0
                return (prediction, confidence)
            }
            
        } catch {
            print("예측 실패: \\(error)")
        }
        
        return (nil, nil)
    }
}

// 사용 예제
let classifier = WatermelonClassifier()

// 예시 특징 벡터 (실제로는 오디오에서 추출)
let sampleFeatures: [Float] = Array(repeating: 0.0, count: 30)

let (prediction, confidence) = classifier.predict(features: sampleFeatures)
if let pred = prediction, let conf = confidence {
    print("예측: \\(pred), 신뢰도: \\(String(format: "%.4f", conf))")
}
```

### 3. Objective-C에서 Core ML 모델 사용

```objc
/*
 Objective-C에서 Core ML 모델 사용하기
 */

#import <CoreML/CoreML.h>

@interface WatermelonClassifier : NSObject

@property (nonatomic, strong) MLModel *model;

- (void)loadModel;
- (NSDictionary *)predictWithFeatures:(NSArray<NSNumber *> *)features;

@end

@implementation WatermelonClassifier

- (void)loadModel {
    NSURL *modelURL = [[NSBundle mainBundle] URLForResource:@"svm_model" withExtension:@"mlmodel"];
    
    NSError *error;
    self.model = [MLModel modelWithContentsOfURL:modelURL error:&error];
    
    if (error) {
        NSLog(@"Core ML 모델 로드 실패: %@", error.localizedDescription);
    } else {
        NSLog(@"Core ML 모델 로드 완료");
    }
}

- (NSDictionary *)predictWithFeatures:(NSArray<NSNumber *> *)features {
    if (!self.model) {
        return @{@"error": @"모델이 로드되지 않았습니다."};
    }
    
    if (features.count != 30) {
        return @{@"error": @"특징 벡터는 30차원이어야 합니다."};
    }
    
    NSError *error;
    
    // 입력 데이터 준비
    MLMultiArray *multiArray = [[MLMultiArray alloc] initWithShape:@[@30] dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (error) {
        return @{@"error": error.localizedDescription};
    }
    
    for (NSInteger i = 0; i < features.count; i++) {
        multiArray[i] = features[i];
    }
    
    // 예측 수행
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"audio_features": multiArray} error:&error];
    if (error) {
        return @{@"error": error.localizedDescription};
    }
    
    id<MLFeatureProvider> output = [self.model predictionFromFeatures:input error:&error];
    if (error) {
        return @{@"error": error.localizedDescription};
    }
    
    // 결과 추출
    NSString *prediction = [output featureValueForName:@"watermelon_class"].stringValue;
    NSDictionary *probabilities = [output featureValueForName:@"watermelon_class_proba"].dictionaryValue;
    
    return @{
        @"prediction": prediction ?: @"",
        @"probabilities": probabilities ?: @{},
        @"status": @"success"
    };
}

@end
```

---

## 배치 예측

### 1. 대용량 파일 처리

```python
"""
대용량 오디오 파일 배치를 효율적으로 처리하는 예제
"""

import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class BatchWatermelonClassifier:
    """배치 수박 분류 처리기"""
    
    def __init__(self, model_path="data/models/pickle/svm_model.pkl", max_workers=4):
        self.model = load_watermelon_model(model_path)
        self.extractor = AudioFeatureExtractor()
        self.max_workers = max_workers
    
    def process_single_file(self, audio_path):
        """단일 파일 처리"""
        try:
            features = self.extractor.extract_features(audio_path)
            features = features.reshape(1, -1)
            
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            return {
                'file_path': audio_path,
                'file_name': os.path.basename(audio_path),
                'prediction': prediction,
                'confidence': confidence,
                'watermelon_A_prob': probabilities[0],
                'watermelon_B_prob': probabilities[1],
                'watermelon_C_prob': probabilities[2],
                'status': 'success'
            }
        
        except Exception as e:
            return {
                'file_path': audio_path,
                'file_name': os.path.basename(audio_path),
                'prediction': None,
                'confidence': 0.0,
                'error': str(e),
                'status': 'failed'
            }
    
    def process_directory(self, directory_path, output_csv=None):
        """
        디렉토리의 모든 오디오 파일을 병렬 처리합니다.
        
        Args:
            directory_path (str): 처리할 디렉토리 경로
            output_csv (str): 결과를 저장할 CSV 파일 경로
            
        Returns:
            pd.DataFrame: 처리 결과
        """
        # 오디오 파일 찾기
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
        
        print(f"발견된 오디오 파일: {len(audio_files)}개")
        
        # 병렬 처리
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_file = {
                executor.submit(self.process_single_file, audio_file): audio_file 
                for audio_file in audio_files
            }
            
            # 결과 수집
            for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="처리 중"):
                result = future.result()
                results.append(result)
        
        # DataFrame으로 변환
        df = pd.DataFrame(results)
        
        # 통계 출력
        successful = df[df['status'] == 'success']
        print(f"\\n처리 완료: {len(successful)}/{len(df)} 파일")
        print(f"성공률: {len(successful)/len(df)*100:.1f}%")
        
        if len(successful) > 0:
            print("\\n예측 분포:")
            print(successful['prediction'].value_counts())
            print(f"\\n평균 신뢰도: {successful['confidence'].mean():.4f}")
        
        # CSV 저장
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"결과 저장: {output_csv}")
        
        return df

# 배치 처리 실행
if __name__ == "__main__":
    classifier = BatchWatermelonClassifier(max_workers=8)
    
    # 디렉토리 처리
    results_df = classifier.process_directory(
        "data/test_audio",
        output_csv="results/batch_predictions.csv"
    )
    
    # 결과 분석
    print("\\n=== 상세 분석 ===")
    successful_results = results_df[results_df['status'] == 'success']
    
    # 신뢰도별 분포
    confidence_bins = pd.cut(successful_results['confidence'], 
                           bins=[0, 0.5, 0.7, 0.9, 1.0], 
                           labels=['낮음(<0.5)', '보통(0.5-0.7)', '높음(0.7-0.9)', '매우높음(≥0.9)'])
    
    print("신뢰도 분포:")
    print(confidence_bins.value_counts())
```

### 2. 실시간 모니터링 배치 처리

```python
"""
실시간 모니터링이 포함된 배치 처리
"""

import time
import logging
from datetime import datetime
from src.utils.performance_monitor import PerformanceMonitor

class MonitoredBatchClassifier(BatchWatermelonClassifier):
    """모니터링 기능이 포함된 배치 분류기"""
    
    def __init__(self, model_path, max_workers=4, log_level=logging.INFO):
        super().__init__(model_path, max_workers)
        self.monitor = PerformanceMonitor(enable_monitoring=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_with_monitoring(self, directory_path, output_csv=None, 
                              report_interval=100):
        """
        모니터링과 함께 배치 처리를 수행합니다.
        
        Args:
            directory_path (str): 처리할 디렉토리
            output_csv (str): 결과 CSV 파일
            report_interval (int): 진행 상황 리포트 간격
        """
        start_time = time.time()
        
        # 성능 모니터링 시작
        self.monitor.start_step_monitoring("batch_processing")
        
        try:
            # 파일 찾기
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.m4a']:
                audio_files.extend(glob.glob(
                    os.path.join(directory_path, '**', ext), recursive=True
                ))
            
            total_files = len(audio_files)
            self.logger.info(f"배치 처리 시작: {total_files}개 파일")
            
            # 병렬 처리
            results = []
            processed = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_file, audio_file): audio_file 
                    for audio_file in audio_files
                }
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    if result['status'] == 'failed':
                        failed += 1
                    
                    # 진행 상황 리포트
                    if processed % report_interval == 0:
                        elapsed = time.time() - start_time
                        progress = processed / total_files * 100
                        rate = processed / elapsed
                        eta = (total_files - processed) / rate if rate > 0 else 0
                        
                        self.logger.info(
                            f"진행: {processed}/{total_files} ({progress:.1f}%) | "
                            f"실패: {failed} | 속도: {rate:.1f} files/sec | "
                            f"예상 완료: {eta:.0f}초 후"
                        )
            
            # 최종 결과
            df = pd.DataFrame(results)
            elapsed_total = time.time() - start_time
            
            self.logger.info(f"배치 처리 완료: {elapsed_total:.1f}초")
            self.logger.info(f"총 처리: {processed}개, 실패: {failed}개")
            self.logger.info(f"평균 속도: {processed/elapsed_total:.2f} files/sec")
            
            # 성능 모니터링 종료
            metrics = self.monitor.end_step_monitoring(
                "batch_processing",
                {"total_files": total_files, "failed_files": failed}
            )
            
            # 성능 리포트
            self.logger.info(f"메모리 사용량: {metrics.get('memory_used', 0):.2f} MB")
            self.logger.info(f"CPU 시간: {metrics.get('cpu_time', 0):.2f}초")
            
            # 결과 저장
            if output_csv:
                df.to_csv(output_csv, index=False)
                self.logger.info(f"결과 저장: {output_csv}")
            
            return df, metrics
            
        except Exception as e:
            self.logger.error(f"배치 처리 실패: {e}")
            raise

# 모니터링 배치 처리 실행
if __name__ == "__main__":
    classifier = MonitoredBatchClassifier(
        "data/models/pickle/svm_model.pkl",
        max_workers=6
    )
    
    results_df, performance_metrics = classifier.process_with_monitoring(
        "data/large_test_set",
        output_csv="results/monitored_batch_results.csv",
        report_interval=50
    )
    
    # 성능 분석
    print("\\n=== 성능 분석 ===")
    print(f"처리 시간: {performance_metrics.get('execution_time', 0):.2f}초")
    print(f"메모리 사용: {performance_metrics.get('memory_used', 0):.2f} MB")
    print(f"처리 효율성: {performance_metrics.get('throughput', 0):.2f} files/sec")
```

---

## 실시간 스트리밍

### 1. 오디오 스트림 처리

```python
"""
실시간 오디오 스트림에서 수박 분류
"""

import pyaudio
import numpy as np
import threading
import queue
import time
from collections import deque

class RealTimeWatermelonClassifier:
    """실시간 수박 소리 분류기"""
    
    def __init__(self, model_path="data/models/pickle/svm_model.pkl"):
        self.model = load_watermelon_model(model_path)
        self.extractor = AudioFeatureExtractor()
        
        # 오디오 설정
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paFloat32
        
        # 버퍼 설정
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 2))  # 2초 버퍼
        self.prediction_queue = queue.Queue()
        
        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # 예측 결과 저장
        self.recent_predictions = deque(maxlen=10)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 콜백 함수"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """오디오 녹음 시작"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            self.is_recording = True
            print("실시간 녹음 시작...")
            
        except Exception as e:
            print(f"녹음 시작 실패: {e}")
    
    def stop_recording(self):
        """오디오 녹음 중지"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("녹음 중지")
    
    def process_audio_buffer(self):
        """오디오 버퍼에서 특징 추출 및 예측"""
        if len(self.audio_buffer) < self.sample_rate:  # 최소 1초 필요
            return None
        
        try:
            # 버퍼에서 오디오 데이터 추출
            audio_data = np.array(list(self.audio_buffer))
            
            # 특징 추출 (오디오 데이터를 직접 처리)
            features = self._extract_features_from_array(audio_data)
            features = features.reshape(1, -1)
            
            # 예측
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            result = {
                'timestamp': time.time(),
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'watermelon_A': probabilities[0],
                    'watermelon_B': probabilities[1],
                    'watermelon_C': probabilities[2]
                }
            }
            
            return result
            
        except Exception as e:
            print(f"오디오 처리 실패: {e}")
            return None
    
    def _extract_features_from_array(self, audio_data):
        """오디오 배열에서 직접 특징 추출"""
        # MFCC 추출
        mfcc = np.mean(librosa.feature.mfcc(
            y=audio_data, sr=self.sample_rate, n_mfcc=13
        ).T, axis=0)
        
        # Mel spectrogram 통계
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
        mel_mean = np.mean(mel_spec)
        mel_std = np.std(mel_spec)
        
        # Spectral 특징들
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate
        ))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate
        ))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Chroma 특징
        chroma = np.mean(librosa.feature.chroma_stft(
            y=audio_data, sr=self.sample_rate, n_chroma=12
        ).T, axis=0)
        
        # 모든 특징 결합
        features = np.concatenate([
            mfcc,
            [mel_mean, mel_std, spectral_centroid, spectral_rolloff, zero_crossing_rate],
            chroma
        ])
        
        return features
    
    def prediction_worker(self):
        """별도 스레드에서 예측 수행"""
        while self.is_recording:
            result = self.process_audio_buffer()
            if result:
                self.recent_predictions.append(result)
                self.prediction_queue.put(result)
            time.sleep(0.5)  # 0.5초마다 예측
    
    def start_realtime_classification(self, duration=None):
        """
        실시간 분류 시작
        
        Args:
            duration (float): 녹음 지속 시간 (None이면 무한)
        """
        self.start_recording()
        
        # 예측 스레드 시작
        prediction_thread = threading.Thread(target=self.prediction_worker)
        prediction_thread.start()
        
        start_time = time.time()
        
        try:
            while self.is_recording:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # 예측 결과 출력
                try:
                    result = self.prediction_queue.get(timeout=1.0)
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"예측: {result['prediction']} "
                          f"(신뢰도: {result['confidence']:.3f})")
                    
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\\n사용자 중단")
        
        finally:
            self.stop_recording()
            prediction_thread.join()
    
    def get_prediction_summary(self):
        """최근 예측 결과 요약"""
        if not self.recent_predictions:
            return "예측 결과가 없습니다."
        
        predictions = [p['prediction'] for p in self.recent_predictions]
        confidences = [p['confidence'] for p in self.recent_predictions]
        
        from collections import Counter
        prediction_counts = Counter(predictions)
        avg_confidence = np.mean(confidences)
        
        summary = f"""
=== 최근 예측 요약 ===
예측 분포: {dict(prediction_counts)}
평균 신뢰도: {avg_confidence:.3f}
총 예측 수: {len(predictions)}
"""
        return summary

# 실시간 분류 실행
if __name__ == "__main__":
    print("실시간 수박 소리 분류기")
    print("마이크에 수박을 두드리는 소리를 내세요.")
    print("Ctrl+C로 중단할 수 있습니다.")
    
    classifier = RealTimeWatermelonClassifier()
    
    try:
        # 30초간 실시간 분류
        classifier.start_realtime_classification(duration=30)
        
        # 결과 요약
        print(classifier.get_prediction_summary())
        
    except Exception as e:
        print(f"실시간 분류 실패: {e}")
```

---

## 성능 최적화

### 1. 모델 로딩 최적화

```python
"""
모델 로딩과 예측 성능을 최적화하는 예제
"""

import time
import functools
from threading import Lock

class OptimizedWatermelonClassifier:
    """성능 최적화된 수박 분류기"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, model_path="data/models/pickle/svm_model.pkl"):
        """싱글톤 패턴으로 모델 인스턴스 재사용"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path="data/models/pickle/svm_model.pkl"):
        if not self._initialized:
            print(f"모델 로딩: {model_path}")
            start_time = time.time()
            
            self.model = load_watermelon_model(model_path)
            self.extractor = AudioFeatureExtractor()
            
            # 모델 워밍업 (첫 예측 시간 단축)
            self._warmup_model()
            
            load_time = time.time() - start_time
            print(f"모델 로딩 완료: {load_time:.3f}초")
            
            self._initialized = True
    
    def _warmup_model(self):
        """모델 워밍업 - 더미 데이터로 첫 예측 수행"""
        dummy_features = np.random.randn(1, 30).astype(np.float32)
        _ = self.model.predict(dummy_features)
        _ = self.model.predict_proba(dummy_features)
    
    @functools.lru_cache(maxsize=128)
    def _cached_feature_extraction(self, audio_path_hash, file_mtime):
        """파일 해시와 수정 시간을 이용한 특징 추출 캐싱"""
        return self.extractor.extract_features(audio_path_hash)
    
    def predict_optimized(self, audio_path, use_cache=True):
        """
        최적화된 예측 수행
        
        Args:
            audio_path (str): 오디오 파일 경로
            use_cache (bool): 특징 추출 캐시 사용 여부
        """
        start_time = time.time()
        
        try:
            if use_cache:
                # 파일 메타데이터를 이용한 캐싱
                import os
                file_stat = os.stat(audio_path)
                file_mtime = file_stat.st_mtime
                
                features = self._cached_feature_extraction(audio_path, file_mtime)
            else:
                features = self.extractor.extract_features(audio_path)
            
            features = features.reshape(1, -1)
            
            # 예측 (벡터화된 연산 사용)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': max(probabilities),
                'probabilities': {
                    'watermelon_A': float(probabilities[0]),
                    'watermelon_B': float(probabilities[1]),
                    'watermelon_C': float(probabilities[2])
                },
                'processing_time': processing_time,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'processing_time': time.time() - start_time
            }

# 성능 벤치마킹
def benchmark_classifier(audio_files, iterations=100):
    """분류기 성능 벤치마킹"""
    
    print(f"성능 벤치마크: {len(audio_files)}개 파일 × {iterations}회")
    
    # 최적화된 분류기
    optimized_classifier = OptimizedWatermelonClassifier()
    
    # 일반 분류기 (비교용)
    regular_model = load_watermelon_model()
    regular_extractor = AudioFeatureExtractor()
    
    # 최적화된 분류기 테스트
    optimized_times = []
    start_time = time.time()
    
    for _ in range(iterations):
        for audio_file in audio_files:
            result = optimized_classifier.predict_optimized(audio_file, use_cache=True)
            if result['status'] == 'success':
                optimized_times.append(result['processing_time'])
    
    optimized_total_time = time.time() - start_time
    
    # 일반 분류기 테스트
    regular_times = []
    start_time = time.time()
    
    for _ in range(iterations):        
        for audio_file in audio_files:
            file_start = time.time()
            try:
                features = regular_extractor.extract_features(audio_file)
                features = features.reshape(1, -1)
                _ = regular_model.predict(features)
                _ = regular_model.predict_proba(features)
                regular_times.append(time.time() - file_start)
            except:
                pass
    
    regular_total_time = time.time() - start_time
    
    # 결과 분석
    print("\\n=== 벤치마크 결과 ===")
    print(f"최적화된 분류기:")
    print(f"  총 시간: {optimized_total_time:.3f}초")
    print(f"  평균 예측 시간: {np.mean(optimized_times)*1000:.2f}ms")
    print(f"  처리량: {len(optimized_times)/optimized_total_time:.1f} predictions/sec")
    
    print(f"\\n일반 분류기:")
    print(f"  총 시간: {regular_total_time:.3f}초")
    print(f"  평균 예측 시간: {np.mean(regular_times)*1000:.2f}ms")
    print(f"  처리량: {len(regular_times)/regular_total_time:.1f} predictions/sec")
    
    speedup = regular_total_time / optimized_total_time
    print(f"\\n성능 향상: {speedup:.2f}x 빠름")

# 벤치마크 실행
if __name__ == "__main__":
    test_files = glob.glob("data/raw/test/**/*.wav", recursive=True)[:10]
    benchmark_classifier(test_files, iterations=50)
```

이 문서는 수박 소리 분류 시스템의 모델을 다양한 환경과 용도로 활용하는 포괄적인 예제를 제공합니다. Pickle 모델의 Python 활용부터 Core ML의 모바일 앱 통합, 실시간 처리, 성능 최적화까지 실제 운영 환경에서 필요한 모든 사용 사례를 다루고 있습니다.