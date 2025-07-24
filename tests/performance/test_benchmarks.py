"""
성능 벤치마크 및 메모리 사용량 테스트

전체 파이프라인의 처리 속도, 메모리 효율성, 확장성을 검증하는 포괄적인 성능 테스트 스위트
"""

import os
import sys
import tempfile
import shutil
import time
import psutil
import threading
from typing import Dict, List, Tuple
import pytest
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.audio.feature_extraction import extract_features, AudioFeatureExtractor
from src.data.augmentation import BatchAugmentor, augment_noise
from src.data.pipeline import DataPipeline
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.utils.performance_monitor import PerformanceMonitor
from config import Config


@dataclass
class BenchmarkResult:
    """벤치마크 결과를 저장하는 데이터 클래스"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput: float  # items per second
    memory_efficiency: float  # MB per item
    success_rate: float  # 0.0 - 1.0
    additional_metrics: Dict = None


class PerformanceBenchmark:
    """성능 벤치마크 기본 클래스"""
    
    def __init__(self):
        self.temp_dir = None
        self.performance_monitor = PerformanceMonitor()
        self.results: List[BenchmarkResult] = []
    
    def setup_benchmark(self):
        """벤치마크 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.performance_monitor.start_monitoring()
    
    def teardown_benchmark(self):
        """벤치마크 정리"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.performance_monitor.stop_monitoring()
    
    def measure_performance(self, func, *args, **kwargs) -> Tuple[any, BenchmarkResult]:
        """함수 성능 측정"""
        # 초기 메모리 사용량
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU 사용량 측정 시작
        cpu_percent_start = process.cpu_percent()
        
        # 실행 시간 측정
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            print(f"Benchmark function failed: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 최종 메모리 사용량
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # CPU 사용량 (간단한 근사치)
        cpu_percent = process.cpu_percent()
        
        benchmark_result = BenchmarkResult(
            test_name=func.__name__,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=final_memory,
            cpu_usage_percent=cpu_percent,
            throughput=0.0,  # 호출자가 설정
            memory_efficiency=0.0,  # 호출자가 설정
            success_rate=1.0 if success else 0.0
        )
        
        return result, benchmark_result
    
    def create_test_audio_files(self, count: int, duration: float = 2.0, 
                               sr: int = 22050) -> List[str]:
        """테스트용 오디오 파일들 생성"""
        audio_files = []
        
        for i in range(count):
            # 다양한 주파수의 사인파 생성
            freq = 440 + (i * 50) % 1000  # 440Hz부터 시작해서 변화
            t = np.linspace(0, duration, int(sr * duration))
            amplitude = 0.3 + (i % 5) * 0.1  # 진폭 변화
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            
            # 약간의 노이즈 추가 (실제와 유사하게)
            noise = np.random.normal(0, 0.02, len(signal))
            signal += noise
            
            filename = f"test_audio_{i:04d}.wav"
            filepath = os.path.join(self.temp_dir, filename)
            sf.write(filepath, signal, sr)
            audio_files.append(filepath)
        
        return audio_files


class TestFeatureExtractionBenchmark:
    """특징 추출 성능 벤치마크"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.benchmark = PerformanceBenchmark()
        self.benchmark.setup_benchmark()
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        self.benchmark.teardown_benchmark()
    
    def test_single_file_extraction_benchmark(self):
        """단일 파일 특징 추출 성능 테스트"""
        # 다양한 길이의 파일로 테스트
        durations = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        for duration in durations:
            audio_files = self.benchmark.create_test_audio_files(1, duration)
            audio_file = audio_files[0]
            
            def extract_single():
                return extract_features(audio_file)
            
            features, result = self.benchmark.measure_performance(extract_single)
            
            if features is not None:
                result.throughput = duration / result.execution_time  # seconds per second
                result.memory_efficiency = result.memory_usage_mb / duration
                
                # 성능 기준 검증
                assert result.execution_time < duration * 0.5, \
                    f"Feature extraction too slow: {result.execution_time:.2f}s for {duration}s audio"
                
                assert result.memory_usage_mb < 100, \
                    f"Memory usage too high: {result.memory_usage_mb:.2f}MB"
                
                print(f"Duration {duration}s: {result.execution_time:.3f}s, "
                      f"{result.memory_usage_mb:.1f}MB, throughput: {result.throughput:.1f}x")
    
    def test_batch_extraction_benchmark(self):
        """배치 특징 추출 성능 테스트"""
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            audio_files = self.benchmark.create_test_audio_files(batch_size, 2.0)
            
            def extract_batch():
                results = []
                for audio_file in audio_files:
                    features = extract_features(audio_file)
                    if features is not None:
                        results.append(features)
                return results
            
            features_list, result = self.benchmark.measure_performance(extract_batch)
            
            if features_list:
                result.throughput = len(features_list) / result.execution_time
                result.memory_efficiency = result.memory_usage_mb / len(features_list)
                
                # 성능 기준 검증
                max_time_per_item = 0.5  # 파일당 최대 0.5초
                assert result.execution_time < batch_size * max_time_per_item, \
                    f"Batch extraction too slow: {result.execution_time:.2f}s for {batch_size} files"
                
                print(f"Batch {batch_size}: {result.execution_time:.3f}s, "
                      f"{result.throughput:.1f} files/s, {result.memory_efficiency:.2f}MB/file")
    
    def test_concurrent_extraction_benchmark(self):
        """동시 특징 추출 성능 테스트"""
        num_files = 20
        num_workers = 4
        
        audio_files = self.benchmark.create_test_audio_files(num_files, 2.0)
        
        def extract_concurrent():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(extract_features, f) for f in audio_files]
                results = [f.result() for f in futures if f.result() is not None]
            return results
        
        features_list, result = self.benchmark.measure_performance(extract_concurrent)
        
        if features_list:
            result.throughput = len(features_list) / result.execution_time
            result.memory_efficiency = result.memory_usage_mb / len(features_list)
            
            # 동시 처리가 순차 처리보다 빨라야 함
            print(f"Concurrent extraction: {result.execution_time:.3f}s, "
                  f"{result.throughput:.1f} files/s, {num_workers} workers")


class TestAugmentationBenchmark:
    """데이터 증강 성능 벤치마크"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.benchmark = PerformanceBenchmark()
        self.benchmark.setup_benchmark()
        
        # 소음 파일들 생성
        self.noise_files = []
        for i in range(3):
            noise_data = np.random.normal(0, 0.1, 22050 * 2)  # 2초 소음
            noise_file = os.path.join(self.benchmark.temp_dir, f"noise_{i}.wav")
            sf.write(noise_file, noise_data, 22050)
            self.noise_files.append(noise_file)
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        self.benchmark.teardown_benchmark()
    
    def test_single_augmentation_benchmark(self):
        """단일 증강 성능 테스트"""
        audio_files = self.benchmark.create_test_audio_files(1, 5.0)  # 5초 파일
        audio_file = audio_files[0]
        
        snr_levels = [-5, 0, 5, 10]
        
        def augment_single():
            results = []
            for noise_file in self.noise_files:
                for snr in snr_levels:
                    result_file = augment_noise(
                        audio_file, noise_file, snr, self.benchmark.temp_dir)
                    if result_file:
                        results.append(result_file)
            return results
        
        augmented_files, result = self.benchmark.measure_performance(augment_single)
        
        if augmented_files:
            expected_count = len(self.noise_files) * len(snr_levels)
            result.throughput = len(augmented_files) / result.execution_time
            result.memory_efficiency = result.memory_usage_mb / len(augmented_files)
            
            # 모든 조합이 생성되어야 함
            assert len(augmented_files) == expected_count
            
            print(f"Single augmentation: {result.execution_time:.3f}s, "
                  f"{len(augmented_files)} files, {result.throughput:.1f} files/s")
    
    def test_batch_augmentation_benchmark(self):
        """배치 증강 성능 테스트"""
        num_files = 10
        audio_files = self.benchmark.create_test_audio_files(num_files, 2.0)
        
        # 오디오 디렉토리 구조 생성
        audio_dir = os.path.join(self.benchmark.temp_dir, "audio")
        noise_dir = os.path.join(self.benchmark.temp_dir, "noise")
        output_dir = os.path.join(self.benchmark.temp_dir, "output")
        
        os.makedirs(audio_dir)
        os.makedirs(noise_dir)
        os.makedirs(output_dir)
        
        # 파일들 이동
        for i, audio_file in enumerate(audio_files):
            new_path = os.path.join(audio_dir, f"audio_{i}.wav")
            os.rename(audio_file, new_path)
        
        for i, noise_file in enumerate(self.noise_files):
            new_path = os.path.join(noise_dir, f"noise_{i}.wav")
            shutil.copy2(noise_file, new_path)
        
        def augment_batch():
            batch_augmentor = BatchAugmentor()
            return batch_augmentor.augment_class_directory(
                audio_dir, noise_dir, output_dir, [0, 5, 10])
        
        results, result = self.benchmark.measure_performance(augment_batch)
        
        if results:
            total_augmented = sum(r.total_created for r in results)
            result.throughput = total_augmented / result.execution_time
            result.memory_efficiency = result.memory_usage_mb / total_augmented if total_augmented > 0 else 0
            
            print(f"Batch augmentation: {result.execution_time:.3f}s, "
                  f"{total_augmented} files, {result.throughput:.1f} files/s")


class TestPipelineBenchmark:
    """전체 파이프라인 성능 벤치마크"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.benchmark = PerformanceBenchmark()
        self.benchmark.setup_benchmark()
        
        # 테스트 데이터 구조 생성
        self.data_dir = os.path.join(self.benchmark.temp_dir, "data")
        self.create_test_dataset()
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        self.benchmark.teardown_benchmark()
    
    def create_test_dataset(self):
        """테스트 데이터셋 생성"""
        # 디렉토리 구조 생성
        raw_dir = os.path.join(self.data_dir, "raw")
        noise_dir = os.path.join(self.data_dir, "noise")
        
        for split in ["train", "validation", "test"]:
            for class_name in ["watermelon_A", "watermelon_B", "watermelon_C"]:
                class_dir = os.path.join(raw_dir, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # 각 클래스에 파일 생성 (클래스별로 다른 특성)
                num_files = 10 if split == "train" else 3
                for i in range(num_files):
                    freq = 400 + hash(class_name) % 200
                    t = np.linspace(0, 1.5, int(22050 * 1.5))
                    signal = 0.4 * np.sin(2 * np.pi * freq * t)
                    
                    # 클래스별 특성 추가
                    if "A" in class_name:
                        signal += 0.1 * np.sin(2 * np.pi * 880 * t)  # 고주파 성분
                    elif "B" in class_name:
                        signal += 0.1 * np.sin(2 * np.pi * 220 * t)  # 저주파 성분
                    
                    filename = f"{class_name}_{split}_{i:02d}.wav"
                    filepath = os.path.join(class_dir, filename)
                    sf.write(filepath, signal, 22050)
        
        # 소음 파일 생성
        os.makedirs(noise_dir, exist_ok=True)
        for i in range(2):
            noise = np.random.normal(0, 0.08, int(22050 * 1.5))
            filename = f"noise_{i}.wav"
            filepath = os.path.join(noise_dir, filename)
            sf.write(filepath, noise, 22050)
    
    def test_full_pipeline_benchmark(self):
        """전체 파이프라인 성능 테스트"""
        config = Config(
            data_directory=self.data_dir,
            noise_directory=os.path.join(self.data_dir, "noise"),
            snr_levels=[0, 5],
            augmentation_factor=1
        )
        
        def run_full_pipeline():
            # 1. 데이터 파이프라인
            pipeline = DataPipeline(config)
            dataset_split = pipeline.run_complete_pipeline()
            
            # 2. 모델 훈련 (빠른 설정)
            trainer = ModelTrainer(config)
            
            # 작은 하이퍼파라미터 그리드로 빠른 테스트
            quick_config = {
                'svm': {
                    'C': [1.0],
                    'gamma': ['scale']
                },
                'random_forest': {
                    'n_estimators': [50],
                    'max_depth': [5]
                }
            }
            trainer.model_configs = quick_config
            
            training_results = trainer.train_models(
                dataset_split.train_features,
                dataset_split.train_labels,
                cv_folds=3  # 빠른 교차 검증
            )
            
            # 3. 모델 평가
            evaluator = ModelEvaluator()
            evaluation_results = {}
            
            for model_name, training_result in training_results.items():
                eval_result = evaluator.evaluate_model(
                    training_result.best_model,
                    dataset_split.test_features,
                    dataset_split.test_labels,
                    model_name
                )
                evaluation_results[model_name] = eval_result
            
            return {
                'dataset_split': dataset_split,
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
        
        results, benchmark_result = self.benchmark.measure_performance(run_full_pipeline)
        
        if results:
            # 처리된 파일 수 계산
            total_files = len(results['dataset_split'].train_files) + \
                         len(results['dataset_split'].validation_files) + \
                         len(results['dataset_split'].test_files)
            
            benchmark_result.throughput = total_files / benchmark_result.execution_time
            benchmark_result.memory_efficiency = benchmark_result.memory_usage_mb / total_files
            
            # 결과 검증
            assert len(results['training_results']) >= 2  # SVM, RF
            assert len(results['evaluation_results']) >= 2
            
            print(f"Full pipeline: {benchmark_result.execution_time:.1f}s, "
                  f"{total_files} files, Peak memory: {benchmark_result.peak_memory_mb:.1f}MB")
            
            # 성능 기준 (조정 가능)
            assert benchmark_result.execution_time < 60, \
                f"Pipeline too slow: {benchmark_result.execution_time:.1f}s"
            
            assert benchmark_result.peak_memory_mb < 500, \
                f"Memory usage too high: {benchmark_result.peak_memory_mb:.1f}MB"


class TestMemoryBenchmark:
    """메모리 사용량 특화 벤치마크"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.benchmark = PerformanceBenchmark()
        self.benchmark.setup_benchmark()
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        self.benchmark.teardown_benchmark()
    
    def test_memory_leak_detection(self):
        """메모리 누수 감지 테스트"""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # 반복적으로 특징 추출 수행
        for i in range(10):
            audio_files = self.benchmark.create_test_audio_files(5, 2.0)
            
            for audio_file in audio_files:
                features = extract_features(audio_file)
                del features  # 명시적 삭제
            
            # 파일 정리
            for audio_file in audio_files:
                os.remove(audio_file)
            
            # 가비지 컬렉션 수행
            gc.collect()
        
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # 객체 증가가 합리적인 범위 내여야 함
        assert object_growth < 1000, \
            f"Potential memory leak detected: {object_growth} new objects"
        
        print(f"Object growth after 10 iterations: {object_growth}")
    
    def test_large_dataset_memory_usage(self):
        """큰 데이터셋 메모리 사용량 테스트"""
        # 많은 수의 짧은 파일 vs 적은 수의 긴 파일
        test_scenarios = [
            ("many_short", 100, 1.0),  # 100개의 1초 파일
            ("few_long", 10, 10.0),    # 10개의 10초 파일
        ]
        
        for scenario_name, num_files, duration in test_scenarios:
            audio_files = self.benchmark.create_test_audio_files(num_files, duration)
            
            def extract_all_features():
                features_list = []
                for audio_file in audio_files:
                    features = extract_features(audio_file)
                    if features:
                        features_list.append(features.to_array())
                return np.array(features_list)
            
            feature_matrix, result = self.benchmark.measure_performance(extract_all_features)
            
            if feature_matrix is not None:
                data_size_mb = feature_matrix.nbytes / 1024 / 1024
                memory_overhead = result.memory_usage_mb - data_size_mb
                
                print(f"{scenario_name}: {result.memory_usage_mb:.1f}MB total, "
                      f"{data_size_mb:.1f}MB data, {memory_overhead:.1f}MB overhead")
                
                # 메모리 오버헤드가 데이터 크기의 2배를 넘지 않아야 함
                assert memory_overhead < data_size_mb * 2, \
                    f"Memory overhead too high: {memory_overhead:.1f}MB"


class TestScalabilityBenchmark:
    """확장성 벤치마크"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.benchmark = PerformanceBenchmark()
        self.benchmark.setup_benchmark()
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        self.benchmark.teardown_benchmark()
    
    def test_scalability_with_file_count(self):
        """파일 수에 따른 확장성 테스트"""
        file_counts = [10, 25, 50, 100]
        results = []
        
        for count in file_counts:
            audio_files = self.benchmark.create_test_audio_files(count, 1.0)
            
            def process_files():
                feature_list = []
                for audio_file in audio_files:
                    features = extract_features(audio_file)
                    if features:
                        feature_list.append(features)
                return feature_list
            
            features, result = self.benchmark.measure_performance(process_files)
            
            if features:
                result.throughput = len(features) / result.execution_time
                results.append((count, result))
                
                print(f"{count} files: {result.execution_time:.2f}s, "
                      f"{result.throughput:.1f} files/s, {result.memory_usage_mb:.1f}MB")
        
        # 선형 확장성 검증 (처리량이 크게 떨어지지 않아야 함)
        if len(results) >= 2:
            first_throughput = results[0][1].throughput
            last_throughput = results[-1][1].throughput
            
            # 처리량이 50% 이상 떨어지면 확장성 문제
            throughput_ratio = last_throughput / first_throughput
            assert throughput_ratio > 0.5, \
                f"Poor scalability: throughput dropped to {throughput_ratio:.1%}"
    
    def test_concurrent_processing_scalability(self):
        """동시 처리 확장성 테스트"""
        num_files = 40
        worker_counts = [1, 2, 4, 8]
        
        audio_files = self.benchmark.create_test_audio_files(num_files, 1.0)
        results = []
        
        for num_workers in worker_counts:
            def process_concurrent():
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(extract_features, f) for f in audio_files]
                    return [f.result() for f in futures if f.result() is not None]
            
            features, result = self.benchmark.measure_performance(process_concurrent)
            
            if features:
                result.throughput = len(features) / result.execution_time
                results.append((num_workers, result))
                
                print(f"{num_workers} workers: {result.execution_time:.2f}s, "
                      f"{result.throughput:.1f} files/s")
        
        # 멀티스레딩 효과 검증
        if len(results) >= 2:
            single_thread_time = results[0][1].execution_time
            multi_thread_time = results[-1][1].execution_time
            
            speedup = single_thread_time / multi_thread_time
            print(f"Speedup with {worker_counts[-1]} workers: {speedup:.2f}x")
            
            # 최소한의 병렬화 효과가 있어야 함
            assert speedup > 1.2, f"Poor parallelization: {speedup:.2f}x speedup"


# 벤치마크 실행 및 리포트 생성
def run_all_benchmarks():
    """모든 벤치마크 실행 및 결과 저장"""
    import json
    from datetime import datetime
    
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': sys.version,
        },
        'results': {}
    }
    
    # 각 벤치마크 클래스 실행
    benchmark_classes = [
        TestFeatureExtractionBenchmark,
        TestAugmentationBenchmark,
        TestPipelineBenchmark,
        TestMemoryBenchmark,
        TestScalabilityBenchmark
    ]
    
    for benchmark_class in benchmark_classes:
        class_name = benchmark_class.__name__
        print(f"\n=== Running {class_name} ===")
        
        try:
            # pytest를 프로그램적으로 실행
            pytest.main([f"tests/performance/test_benchmarks.py::{class_name}", "-v"])
            benchmark_results['results'][class_name] = "completed"
        except Exception as e:
            print(f"Benchmark {class_name} failed: {e}")
            benchmark_results['results'][class_name] = f"failed: {str(e)}"
    
    # 결과를 JSON 파일로 저장
    results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\nBenchmark results saved to: {results_file}")
    return benchmark_results


if __name__ == "__main__":
    # 개별 테스트 실행 시
    if len(sys.argv) > 1 and sys.argv[1] == "--run-all":
        run_all_benchmarks()
    else:
        # pytest 실행
        pytest.main([__file__, "-v"])