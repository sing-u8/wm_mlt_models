# Core ML 모델 사용 가이드

수박 소리 분류 시스템에서 생성된 Core ML 모델을 사용하는 방법을 설명합니다.

## 개요

Core ML은 Apple에서 개발한 머신러닝 프레임워크로, iOS, macOS, watchOS, tvOS 앱에서 머신러닝 모델을 효율적으로 사용할 수 있게 해줍니다. 본 시스템에서는 scikit-learn으로 훈련된 모델을 Core ML 형식(.mlmodel)으로 변환하여 Apple 생태계에서 사용할 수 있도록 합니다.

## 모델 정보

### 입력 사양
- **입력 이름**: `audio_features`
- **데이터 타입**: Float32
- **형태**: (30,) - 30차원 특징 벡터
- **설명**: 수박 소리에서 추출된 오디오 특징

### 출력 사양
- **클래스 예측**: `watermelon_class`
  - 데이터 타입: String
  - 가능한 값: `"watermelon_A"`, `"watermelon_B"`, `"watermelon_C"`
- **확률 예측**: `watermelon_class_proba`  
  - 데이터 타입: Dictionary
  - 각 클래스별 예측 확률 (0.0 ~ 1.0)

### 특징 벡터 구성 (30차원)
1. **MFCC**: 13개 계수 (mel-frequency cepstral coefficients)
2. **Mel Spectrogram 통계**: 평균, 표준편차 (2개)
3. **Spectral 특징**: Centroid, Rolloff, Zero Crossing Rate (3개)
4. **Chroma Features**: 12차원 피치 클래스 프로파일

## Python에서 Core ML 모델 사용

### 1. 필요한 라이브러리 설치

```bash
pip install coremltools numpy librosa
```

### 2. 모델 로드 및 예측

```python
import coremltools as ct
import numpy as np
import librosa

# Core ML 모델 로드
model = ct.models.MLModel('random_forest_model.mlmodel')

# 오디오 파일에서 특징 추출 함수 (예시)
def extract_features(audio_path):
    """오디오 파일에서 30차원 특징 벡터를 추출합니다."""
    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=22050)
    
    # MFCC 추출 (13개)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Mel Spectrogram 통계 (2개)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512)
    mel_mean = np.mean(mel_spec)
    mel_std = np.std(mel_spec)
    
    # Spectral 특징 (3개)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512))
    
    # Chroma Features (12개)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)
    
    # 특징 벡터 결합 (30차원)
    features = np.concatenate([
        mfcc_mean,                    # 13개
        [mel_mean, mel_std],          # 2개
        [spectral_centroid, spectral_rolloff, zero_crossing_rate],  # 3개
        chroma_mean                   # 12개
    ])
    
    return features.astype(np.float32)

# 오디오 파일 분류
audio_file = "watermelon_sample.wav"
features = extract_features(audio_file)

# Core ML 모델로 예측
input_dict = {'audio_features': features.reshape(1, -1)}
prediction = model.predict(input_dict)

print(f"예측 클래스: {prediction['watermelon_class']}")
print(f"예측 확률: {prediction['watermelon_class_proba']}")
```

### 3. 배치 예측

```python
import os
import pandas as pd

def predict_batch(model, audio_files):
    """여러 오디오 파일을 배치로 예측합니다."""
    results = []
    
    for audio_file in audio_files:
        try:
            # 특징 추출
            features = extract_features(audio_file)
            
            # 예측
            input_dict = {'audio_features': features.reshape(1, -1)}
            prediction = model.predict(input_dict)
            
            results.append({
                'file': os.path.basename(audio_file),
                'predicted_class': prediction['watermelon_class'],
                'confidence': max(prediction['watermelon_class_proba'].values()),
                'probabilities': prediction['watermelon_class_proba']
            })
            
        except Exception as e:
            print(f"오류 발생 {audio_file}: {e}")
            results.append({
                'file': os.path.basename(audio_file),
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'probabilities': {}
            })
    
    return pd.DataFrame(results)

# 사용 예시
audio_files = ['sample1.wav', 'sample2.wav', 'sample3.wav']
results_df = predict_batch(model, audio_files)
print(results_df)
```

## iOS/macOS 앱에서 사용

### 1. Xcode 프로젝트에 모델 추가

1. Xcode에서 프로젝트를 엽니다
2. `random_forest_model.mlmodel` 파일을 프로젝트에 드래그 앤 드롭합니다
3. "Copy items if needed" 체크박스를 선택합니다

### 2. Swift 코드에서 모델 사용

```swift
import CoreML
import AVFoundation

class WatermelonClassifier {
    private var model: random_forest_model?
    
    init() {
        // 모델 로드
        do {
            model = try random_forest_model(configuration: MLModelConfiguration())
        } catch {
            print("모델 로드 실패: \(error)")
        }
    }
    
    func classifyWatermelon(features: [Float]) -> (class: String, confidence: Double)? {
        guard let model = model else { return nil }
        guard features.count == 30 else {
            print("특징 벡터는 30차원이어야 합니다")
            return nil
        }
        
        do {
            // 입력 데이터 준비
            let multiArray = try MLMultiArray(shape: [30], dataType: .float32)
            for (index, value) in features.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }
            
            let input = random_forest_modelInput(audio_features: multiArray)
            
            // 예측 수행
            let prediction = try model.prediction(input: input)
            
            // 최고 확률 클래스 찾기
            let maxProbability = prediction.watermelon_class_proba.max { $0.value < $1.value }
            let confidence = maxProbability?.value ?? 0.0
            
            return (class: prediction.watermelon_class, confidence: confidence)
            
        } catch {
            print("예측 실패: \(error)")
            return nil
        }
    }
}

// 사용 예시
let classifier = WatermelonClassifier()
let features: [Float] = Array(repeating: 0.5, count: 30) // 실제 특징 벡터로 교체

if let result = classifier.classifyWatermelon(features: features) {
    print("예측 클래스: \(result.class)")
    print("신뢰도: \(result.confidence)")
}
```

### 3. 오디오 특징 추출 (iOS)

```swift
import AVFoundation
import Accelerate

class AudioFeatureExtractor {
    
    func extractFeatures(from audioURL: URL) -> [Float]? {
        // 오디오 파일 로드
        guard let audioFile = try? AVAudioFile(forReading: audioURL) else {
            print("오디오 파일 로드 실패")
            return nil
        }
        
        let format = audioFile.processingFormat
        let frameCount = UInt32(audioFile.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        
        do {
            try audioFile.read(into: buffer)
        } catch {
            print("오디오 읽기 실패: \(error)")
            return nil
        }
        
        guard let channelData = buffer.floatChannelData?[0] else {
            return nil
        }
        
        // 여기서 실제 특징 추출 알고리즘을 구현해야 합니다
        // 예시로 간단한 통계적 특징만 계산
        let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(frameCount)))
        
        // 실제 구현에서는 MFCC, Mel Spectrogram 등을 계산해야 합니다
        // 이는 복잡한 DSP 작업이므로, Python에서 미리 계산하거나
        // 별도의 오디오 처리 라이브러리를 사용하는 것이 좋습니다
        
        return extractSimpleFeatures(from: samples)
    }
    
    private func extractSimpleFeatures(from samples: [Float]) -> [Float] {
        // 간단한 통계적 특징 30개 추출 (예시)
        var features = [Float]()
        
        let mean = samples.reduce(0, +) / Float(samples.count)
        let variance = samples.map { pow($0 - mean, 2) }.reduce(0, +) / Float(samples.count)
        let std = sqrt(variance)
        
        // 30개 특징으로 확장 (실제로는 MFCC, Chroma 등을 계산해야 함)
        for i in 0..<30 {
            switch i % 3 {
            case 0: features.append(mean)
            case 1: features.append(std)
            default: features.append(Float.random(in: -1...1)) // 예시
            }
        }
        
        return features
    }
}
```

## 성능 최적화

### 1. 모델 양자화
Core ML 모델의 크기를 줄이고 추론 속도를 향상시키려면 양자화를 사용할 수 있습니다:

```python
import coremltools as ct

# 모델 로드
model = ct.models.MLModel('random_forest_model.mlmodel')

# 양자화 적용 (16-bit 정밀도)
model_quantized = ct.models.neural_network.quantization_utils.quantize_weights(
    model, nbits=16
)

# 양자화된 모델 저장
model_quantized.save('random_forest_model_quantized.mlmodel')
```

### 2. 배치 예측 최적화
여러 샘플을 한 번에 처리할 때는 배치 크기를 조정하여 성능을 향상시킬 수 있습니다.

### 3. 메모리 관리
iOS/macOS 앱에서는 메모리 사용량을 주의깊게 관리해야 합니다:

```swift
// 모델을 필요할 때만 로드하고 사용 후 해제
class WatermelonClassifierManager {
    private var model: random_forest_model?
    
    func loadModel() {
        if model == nil {
            model = try? random_forest_model(configuration: MLModelConfiguration())
        }
    }
    
    func unloadModel() {
        model = nil
    }
}
```

## 문제 해결

### 일반적인 문제

1. **모델 로드 실패**
   - Core ML 모델 파일이 프로젝트에 올바르게 추가되었는지 확인
   - iOS 버전 호환성 확인 (Core ML은 iOS 11.0+ 필요)

2. **예측 결과 불일치**
   - 입력 특징 벡터의 정규화가 올바른지 확인
   - 특징 추출 과정이 훈련 시와 동일한지 확인

3. **성능 문제**
   - 모델 양자화 적용 고려
   - 특징 추출 과정 최적화
   - 필요시 백그라운드 스레드에서 실행

### 디버깅 팁

1. **입력 데이터 확인**
```python
# 특징 벡터 범위 확인
print(f"특징 최솟값: {features.min()}")
print(f"특징 최댓값: {features.max()}")
print(f"특징 평균: {features.mean()}")
print(f"특징 형태: {features.shape}")
```

2. **예측 결과 분석**
```python
# 모든 클래스의 확률 출력
for class_name, probability in prediction['watermelon_class_proba'].items():
    print(f"{class_name}: {probability:.4f}")
```

## 추가 리소스

- [Core ML 공식 문서](https://developer.apple.com/documentation/coreml)
- [coremltools 가이드](https://coremltools.readme.io/docs)
- [iOS 머신러닝 튜토리얼](https://developer.apple.com/machine-learning/)

## 지원 및 문의

모델 사용 중 문제가 발생하면 다음 정보를 포함하여 문의해 주세요:

1. 사용 중인 플랫폼 (iOS/macOS/Python)
2. 오류 메시지 또는 예상과 다른 결과
3. 입력 데이터의 형태와 범위
4. 사용 중인 라이브러리 버전

---

이 가이드는 수박 소리 분류 시스템의 Core ML 모델 사용법을 다룹니다. 실제 앱 개발 시에는 사용자 인터페이스, 오디오 녹음, 실시간 처리 등 추가적인 구현이 필요할 수 있습니다.