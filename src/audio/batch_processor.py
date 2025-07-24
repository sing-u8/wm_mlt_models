"""
배치 특징 추출 최적화 모듈

대용량 오디오 데이터의 효율적인 배치 처리를 위한 최적화된 특징 추출 시스템
"""

import os
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Iterator
from pathlib import Path
import numpy as np
import librosa
from dataclasses import dataclass
import time
import psutil
from functools import partial
import logging

from .feature_extraction import extract_features, AudioFeatureExtractor, FeatureVector
from ..utils.logger import LoggerMixin
from ..utils.performance_monitor import PerformanceMonitor
from config import DEFAULT_CONFIG


@dataclass
class BatchProcessingConfig:
    """배치 처리 설정"""
    max_workers: Optional[int] = None  # None이면 CPU 코어 수
    chunk_size: int = 32  # 한 번에 처리할 파일 수
    memory_limit_gb: float = 4.0  # 메모리 사용 제한 (GB)
    use_multiprocessing: bool = True  # True: 프로세스, False: 스레드
    prefetch_count: int = 2  # 미리 로드할 청크 수
    cache_features: bool = False  # 특징 캐싱 여부
    progress_callback: Optional[callable] = None  # 진행률 콜백


@dataclass
class BatchResult:
    """배치 처리 결과"""
    features: List[np.ndarray]
    file_paths: List[str]
    processing_times: List[float]
    total_time: float
    success_count: int
    failure_count: int
    memory_peak_mb: float
    errors: List[Tuple[str, str]]  # (파일 경로, 에러 메시지)


class OptimizedFeatureExtractor(AudioFeatureExtractor, LoggerMixin):
    """최적화된 특징 추출기"""
    
    def __init__(self, config=None):
        super().__init__(config or DEFAULT_CONFIG)
        self.logger = self.get_logger()
        
        # 메모리 효율을 위한 librosa 설정
        self._setup_librosa_optimization()
    
    def _setup_librosa_optimization(self):
        """librosa 최적화 설정"""
        # librosa의 캐시 크기 제한
        try:
            import librosa.cache
            librosa.cache.MAX_CACHE_SIZE = 50  # 50MB로 제한
        except Exception as e:
            self.logger.warning(f"librosa 캐시 설정 실패: {e}")
    
    def extract_features_optimized(self, audio_file: str, 
                                 return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        최적화된 특징 추출
        
        Args:
            audio_file: 오디오 파일 경로
            return_metadata: 메타데이터 반환 여부
            
        Returns:
            특징 벡터 또는 (특징 벡터, 메타데이터) 튜플
        """
        start_time = time.time()
        metadata = {}
        
        try:
            # 메모리 효율적인 오디오 로딩
            y, sr = self._load_audio_efficient(audio_file)
            
            if y is None:
                return None if not return_metadata else (None, metadata)
            
            # 특징 추출
            features = self._extract_all_features_optimized(y, sr)
            
            if return_metadata:
                metadata = {
                    'file_path': audio_file,
                    'file_size_mb': os.path.getsize(audio_file) / (1024*1024),
                    'duration': len(y) / sr,
                    'processing_time': time.time() - start_time,
                    'sample_rate': sr
                }
                return features.to_array(), metadata
            
            return features.to_array()
            
        except Exception as e:
            self.logger.error(f"특징 추출 실패 {audio_file}: {e}")
            return None if not return_metadata else (None, metadata)
    
    def _load_audio_efficient(self, audio_file: str) -> Tuple[Optional[np.ndarray], int]:
        """메모리 효율적인 오디오 로딩"""
        try:
            # 메모리 사용량을 줄이기 위해 mono=True, dtype=np.float32 사용
            y, sr = librosa.load(
                audio_file, 
                sr=self.sr, 
                mono=True, 
                dtype=np.float32,
                res_type='kaiser_fast'  # 빠른 리샘플링
            )
            
            # 무음 파일 체크
            if np.max(np.abs(y)) < 1e-6:
                self.logger.warning(f"무음 파일 감지: {audio_file}")
            
            return y, sr
            
        except Exception as e:
            self.logger.error(f"오디오 로딩 실패 {audio_file}: {e}")
            return None, 0
    
    def _extract_all_features_optimized(self, y: np.ndarray, sr: int) -> FeatureVector:
        """최적화된 모든 특징 추출"""
        # 한 번의 스펙트로그램 계산으로 여러 특징 추출
        stft = librosa.stft(y, hop_length=self.hop_length, dtype=np.complex64)  # float32 사용
        magnitude = np.abs(stft)
        
        # MFCC 추출 (미리 계산된 magnitude 사용)
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(magnitude**2),
            sr=sr,
            n_mfcc=self.n_mfcc,
            dtype=np.float32
        ).mean(axis=1)
        
        # Mel spectrogram 통계 (미리 계산된 magnitude 사용)
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude**2,
            sr=sr,
            dtype=np.float32
        )
        mel_mean = np.mean(mel_spec)
        mel_std = np.std(mel_spec)
        
        # Spectral features (미리 계산된 magnitude 사용)
        spectral_centroids = librosa.feature.spectral_centroid(
            S=magnitude, sr=sr, dtype=np.float32
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=sr, dtype=np.float32
        )
        
        # Zero crossing rate (원본 신호 사용)
        zcr = librosa.feature.zero_crossing_rate(y, dtype=np.float32)
        
        # Chroma features (미리 계산된 stft 사용)
        chroma = librosa.feature.chroma_stft(
            S=magnitude, sr=sr, n_chroma=self.n_chroma, dtype=np.float32
        ).mean(axis=1)
        
        return FeatureVector(
            mfcc=mfcc.astype(np.float32),
            mel_mean=float(mel_mean),
            mel_std=float(mel_std),
            spectral_centroid=float(spectral_centroids.mean()),
            spectral_rolloff=float(spectral_rolloff.mean()),
            zero_crossing_rate=float(zcr.mean()),
            chroma=chroma.astype(np.float32)
        )


class BatchFeatureProcessor(LoggerMixin):
    """배치 특징 추출 프로세서"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        self.logger = self.get_logger()
        self.performance_monitor = PerformanceMonitor()
        
        # 최적화된 특징 추출기 생성
        self.extractor = OptimizedFeatureExtractor()
        
        # 워커 수 자동 설정
        if self.config.max_workers is None:
            cpu_count = mp.cpu_count()
            # 메모리 제한을 고려한 워커 수 조정
            memory_gb = psutil.virtual_memory().total / (1024**3)
            max_workers_by_memory = max(1, int(memory_gb / self.config.memory_limit_gb * cpu_count))
            self.config.max_workers = min(cpu_count, max_workers_by_memory)
        
        self.logger.info(f"배치 프로세서 초기화: {self.config.max_workers}개 워커, "
                        f"청크 크기: {self.config.chunk_size}")
    
    def process_files(self, file_paths: List[str]) -> BatchResult:
        """
        파일 목록을 배치로 처리
        
        Args:
            file_paths: 처리할 오디오 파일 경로 목록
            
        Returns:
            배치 처리 결과
        """
        self.logger.info(f"배치 처리 시작: {len(file_paths)}개 파일")
        start_time = time.time()
        
        # 성능 모니터링 시작
        self.performance_monitor.start_monitoring()
        
        try:
            if self.config.use_multiprocessing:
                result = self._process_with_multiprocessing(file_paths)
            else:
                result = self._process_with_threading(file_paths)
        finally:
            # 성능 모니터링 종료
            perf_stats = self.performance_monitor.stop_monitoring()
        
        total_time = time.time() - start_time
        
        # 결과 정리
        batch_result = BatchResult(
            features=result['features'],
            file_paths=result['file_paths'], 
            processing_times=result['processing_times'],
            total_time=total_time,
            success_count=result['success_count'],
            failure_count=result['failure_count'],
            memory_peak_mb=perf_stats.get('peak_memory_mb', 0),
            errors=result['errors']
        )
        
        self.logger.info(f"배치 처리 완료: {batch_result.success_count}/{len(file_paths)} 성공, "
                        f"{total_time:.2f}초, 피크 메모리: {batch_result.memory_peak_mb:.1f}MB")
        
        return batch_result
    
    def _process_with_multiprocessing(self, file_paths: List[str]) -> Dict:
        """멀티프로세싱을 사용한 배치 처리"""
        features = []
        successful_paths = []
        processing_times = []
        errors = []
        success_count = 0
        failure_count = 0
        
        # 청크로 분할
        chunks = [file_paths[i:i + self.config.chunk_size] 
                 for i in range(0, len(file_paths), self.config.chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 각 청크를 병렬로 처리
            future_to_chunk = {
                executor.submit(self._process_chunk_worker, chunk): chunk 
                for chunk in chunks
            }
            
            completed_chunks = 0
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    chunk_result = future.result()
                    
                    features.extend(chunk_result['features'])
                    successful_paths.extend(chunk_result['file_paths'])
                    processing_times.extend(chunk_result['processing_times'])
                    errors.extend(chunk_result['errors'])
                    success_count += chunk_result['success_count']
                    failure_count += chunk_result['failure_count']
                    
                except Exception as e:
                    self.logger.error(f"청크 처리 실패: {e}")
                    failure_count += len(chunk)
                    errors.extend([(path, str(e)) for path in chunk])
                
                completed_chunks += 1
                
                # 진행률 콜백 호출
                if self.config.progress_callback:
                    progress = completed_chunks / len(chunks)
                    self.config.progress_callback(progress)
        
        return {
            'features': features,
            'file_paths': successful_paths,
            'processing_times': processing_times,
            'success_count': success_count,
            'failure_count': failure_count,
            'errors': errors
        }
    
    def _process_with_threading(self, file_paths: List[str]) -> Dict:
        """멀티스레딩을 사용한 배치 처리"""
        features = []
        successful_paths = []
        processing_times = []
        errors = []
        success_count = 0
        failure_count = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 개별 파일을 병렬로 처리
            future_to_path = {
                executor.submit(self.extractor.extract_features_optimized, path, True): path 
                for path in file_paths
            }
            
            completed_files = 0
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                
                try:
                    result = future.result()
                    
                    if result[0] is not None:  # (features, metadata)
                        features.append(result[0])
                        successful_paths.append(file_path)
                        processing_times.append(result[1].get('processing_time', 0))
                        success_count += 1
                    else:
                        failure_count += 1
                        errors.append((file_path, "특징 추출 실패"))
                        
                except Exception as e:
                    failure_count += 1
                    errors.append((file_path, str(e)))
                
                completed_files += 1
                
                # 진행률 콜백 호출
                if self.config.progress_callback:
                    progress = completed_files / len(file_paths)
                    self.config.progress_callback(progress)
        
        return {
            'features': features,
            'file_paths': successful_paths,
            'processing_times': processing_times,
            'success_count': success_count,
            'failure_count': failure_count,
            'errors': errors
        }
    
    @staticmethod
    def _process_chunk_worker(file_chunk: List[str]) -> Dict:
        """워커 프로세스에서 실행되는 청크 처리 함수"""
        extractor = OptimizedFeatureExtractor()
        
        features = []
        successful_paths = []
        processing_times = []
        errors = []
        success_count = 0
        failure_count = 0
        
        for file_path in file_chunk:
            try:
                result = extractor.extract_features_optimized(file_path, return_metadata=True)
                
                if result[0] is not None:  # (features, metadata)
                    features.append(result[0])
                    successful_paths.append(file_path)
                    processing_times.append(result[1].get('processing_time', 0))
                    success_count += 1
                else:
                    failure_count += 1
                    errors.append((file_path, "특징 추출 실패"))
                    
            except Exception as e:
                failure_count += 1
                errors.append((file_path, str(e)))
        
        return {
            'features': features,
            'file_paths': successful_paths,
            'processing_times': processing_times,
            'success_count': success_count,
            'failure_count': failure_count,
            'errors': errors
        }
    
    def get_optimal_chunk_size(self, total_files: int, 
                             avg_file_size_mb: float = 1.0) -> int:
        """최적 청크 크기 계산"""
        # 메모리 제한을 고려한 청크 크기
        memory_per_worker = self.config.memory_limit_gb / self.config.max_workers
        chunk_size_by_memory = max(1, int(memory_per_worker / avg_file_size_mb))
        
        # 전체 파일 수를 고려한 청크 크기
        chunk_size_by_files = max(1, total_files // (self.config.max_workers * 4))
        
        # 더 작은 값 선택 (안전한 쪽)
        optimal_size = min(chunk_size_by_memory, chunk_size_by_files, self.config.chunk_size)
        
        self.logger.info(f"최적 청크 크기 계산: {optimal_size} "
                        f"(메모리 기준: {chunk_size_by_memory}, 파일 기준: {chunk_size_by_files})")
        
        return optimal_size


class StreamingFeatureProcessor(LoggerMixin):
    """스트리밍 특징 추출 프로세서 (메모리 효율성)"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        self.logger = self.get_logger()
        self.extractor = OptimizedFeatureExtractor()
    
    def process_stream(self, file_paths: List[str], 
                      output_callback: callable) -> Iterator[Tuple[np.ndarray, str]]:
        """
        스트리밍 방식으로 특징 추출
        
        Args:
            file_paths: 처리할 파일 경로 목록
            output_callback: 각 결과를 처리할 콜백 함수
            
        Yields:
            (특징 벡터, 파일 경로) 튜플
        """
        self.logger.info(f"스트리밍 처리 시작: {len(file_paths)}개 파일")
        
        processed_count = 0
        
        for file_path in file_paths:
            try:
                features = self.extractor.extract_features_optimized(file_path)
                
                if features is not None:
                    result = (features, file_path)
                    
                    # 콜백 함수 호출
                    if output_callback:
                        output_callback(result)
                    
                    yield result
                
                processed_count += 1
                
                # 진행률 콜백
                if self.config.progress_callback:
                    progress = processed_count / len(file_paths)
                    self.config.progress_callback(progress)
                    
            except Exception as e:
                self.logger.error(f"스트리밍 처리 실패 {file_path}: {e}")
                continue
        
        self.logger.info(f"스트리밍 처리 완료: {processed_count}개 파일")


# 편의 함수들
def extract_features_batch(file_paths: List[str], 
                         config: BatchProcessingConfig = None) -> BatchResult:
    """
    배치 특징 추출 편의 함수
    
    Args:
        file_paths: 처리할 파일 경로 목록
        config: 배치 처리 설정
        
    Returns:
        배치 처리 결과
    """
    processor = BatchFeatureProcessor(config)
    return processor.process_files(file_paths)


def extract_features_streaming(file_paths: List[str], 
                             output_callback: callable = None,
                             config: BatchProcessingConfig = None) -> Iterator[Tuple[np.ndarray, str]]:
    """
    스트리밍 특징 추출 편의 함수
    
    Args:
        file_paths: 처리할 파일 경로 목록
        output_callback: 결과 처리 콜백
        config: 배치 처리 설정
        
    Yields:
        (특징 벡터, 파일 경로) 튜플
    """
    processor = StreamingFeatureProcessor(config)
    yield from processor.process_stream(file_paths, output_callback)


# 사용 예제
if __name__ == "__main__":
    # 간단한 사용 예제
    import glob
    
    # 테스트 파일 찾기
    test_files = glob.glob("data/raw/train/*/*.wav")[:10]  # 처음 10개만
    
    if test_files:
        print(f"테스트 파일 {len(test_files)}개로 배치 처리 테스트")
        
        # 진행률 콜백 함수
        def progress_callback(progress):
            print(f"진행률: {progress:.1%}")
        
        # 배치 처리 설정
        config = BatchProcessingConfig(
            max_workers=4,
            chunk_size=5,
            progress_callback=progress_callback,
            use_multiprocessing=True
        )
        
        # 배치 처리 실행
        result = extract_features_batch(test_files, config)
        
        print(f"처리 결과: {result.success_count}/{len(test_files)} 성공")
        print(f"총 시간: {result.total_time:.2f}초")
        print(f"평균 처리 시간: {np.mean(result.processing_times):.3f}초/파일")
        print(f"피크 메모리: {result.memory_peak_mb:.1f}MB")
        
        if result.errors:
            print(f"에러 {len(result.errors)}개:")
            for file_path, error in result.errors[:3]:  # 처음 3개만 출력
                print(f"  {file_path}: {error}")
    else:
        print("테스트할 파일을 찾을 수 없습니다.")