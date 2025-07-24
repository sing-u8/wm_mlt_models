"""
대용량 데이터셋 메모리 효율 처리 모듈

메모리 제약이 있는 환경에서 대용량 오디오 데이터셋을 효율적으로 처리하기 위한 
스트리밍 및 청크 기반 처리 시스템
"""

import os
import sys
import gc
import psutil
import numpy as np
from typing import List, Dict, Iterator, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import time
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque
import mmap

from ..audio.batch_processor import OptimizedFeatureExtractor, BatchProcessingConfig
from ..data.parallel_augmentor import OptimizedAudioAugmentor, ParallelAugmentationConfig
from ..utils.logger import LoggerMixin
from ..utils.performance_monitor import PerformanceMonitor
from config import DEFAULT_CONFIG


@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    max_memory_gb: float = 2.0  # 최대 메모리 사용량 (GB)
    chunk_memory_mb: float = 256.0  # 청크당 메모리 한계 (MB)
    gc_threshold: float = 0.8  # GC 실행 임계점 (메모리 사용률)
    memory_check_interval: int = 10  # 메모리 체크 간격 (파일 수)
    swap_threshold_gb: float = 0.5  # 스왑 파일 사용 임계점
    temp_dir_size_limit_gb: float = 10.0  # 임시 디렉토리 크기 제한
    prefetch_buffer_size: int = 5  # 미리 로드할 항목 수
    enable_mmap: bool = True  # 메모리 맵 파일 사용 여부


@dataclass
class ProcessingStats:
    """처리 통계 정보"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_size_mb: float = 0.0
    processed_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    processing_time: float = 0.0
    gc_count: int = 0
    temp_files_created: int = 0
    temp_files_cleaned: int = 0


class MemoryMonitor:
    """실시간 메모리 모니터링"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        self.memory_history = deque(maxlen=100)
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """모니터링 중지 및 통계 반환"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if self.memory_history:
            return {
                'peak_memory_mb': max(self.memory_history),
                'avg_memory_mb': sum(self.memory_history) / len(self.memory_history),
                'memory_samples': len(self.memory_history)
            }
        return {'peak_memory_mb': 0, 'avg_memory_mb': 0, 'memory_samples': 0}
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self._monitoring:
            try:
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self.memory_history.append(memory_mb)
                time.sleep(0.5)  # 0.5초마다 체크
            except Exception:
                break
    
    def get_current_memory_mb(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def is_memory_critical(self) -> bool:
        """메모리 사용량이 임계점을 넘었는지 확인"""
        current_mb = self.get_current_memory_mb()
        limit_mb = self.config.max_memory_gb * 1024
        return current_mb > limit_mb * self.config.gc_threshold


class ChunkedFileProcessor(LoggerMixin):
    """청크 기반 파일 처리기"""
    
    def __init__(self, memory_config: MemoryConfig = None):
        self.memory_config = memory_config or MemoryConfig()
        self.logger = self.get_logger()
        self.memory_monitor = MemoryMonitor(self.memory_config)
        self.stats = ProcessingStats()
        
        # 임시 디렉토리 설정
        self.temp_dir = tempfile.mkdtemp(prefix="wm_large_dataset_")
        self.logger.info(f"임시 디렉토리 생성: {self.temp_dir}")
    
    def process_files_chunked(self, file_paths: List[str], 
                            processor_func: Callable,
                            chunk_size: Optional[int] = None) -> Iterator[Dict]:
        """
        파일을 청크 단위로 처리
        
        Args:
            file_paths: 처리할 파일 경로 목록
            processor_func: 각 청크를 처리할 함수
            chunk_size: 청크 크기 (None이면 자동 계산)
            
        Yields:
            각 청크의 처리 결과
        """
        self.stats.total_files = len(file_paths)
        self.stats.total_size_mb = self._calculate_total_size(file_paths)
        
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(file_paths)
        
        self.logger.info(f"청크 처리 시작: {len(file_paths)}개 파일, 청크 크기: {chunk_size}")
        
        # 메모리 모니터링 시작
        self.memory_monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # 파일을 청크로 분할
            chunks = [file_paths[i:i + chunk_size] 
                     for i in range(0, len(file_paths), chunk_size)]
            
            for chunk_idx, chunk in enumerate(chunks):
                # 메모리 체크 및 정리
                if self.memory_monitor.is_memory_critical():
                    self._force_garbage_collection()
                
                self.logger.info(f"청크 {chunk_idx + 1}/{len(chunks)} 처리 중 ({len(chunk)}개 파일)")
                
                try:
                    # 청크 처리
                    chunk_result = processor_func(chunk)
                    chunk_result['chunk_index'] = chunk_idx
                    chunk_result['chunk_size'] = len(chunk)
                    
                    self.stats.processed_files += len(chunk)
                    self.stats.processed_size_mb += self._calculate_chunk_size(chunk)
                    
                    yield chunk_result
                    
                except Exception as e:
                    self.logger.error(f"청크 {chunk_idx} 처리 실패: {e}")
                    self.stats.failed_files += len(chunk)
                    yield {
                        'chunk_index': chunk_idx,
                        'chunk_size': len(chunk),
                        'error': str(e),
                        'failed_files': chunk
                    }
                
                # 주기적인 가비지 컬렉션
                if (chunk_idx + 1) % 5 == 0:
                    self._force_garbage_collection()
        
        finally:
            # 통계 완료
            self.stats.processing_time = time.time() - start_time
            memory_stats = self.memory_monitor.stop_monitoring()
            self.stats.peak_memory_mb = memory_stats['peak_memory_mb']
            self.stats.avg_memory_mb = memory_stats['avg_memory_mb']
            
            self.logger.info(f"청크 처리 완료: {self.stats.processed_files}/{self.stats.total_files} 성공")
    
    def _calculate_total_size(self, file_paths: List[str]) -> float:
        """전체 파일 크기 계산 (MB)"""
        total_size = 0
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            except Exception:
                continue
        return total_size / (1024 * 1024)
    
    def _calculate_chunk_size(self, chunk: List[str]) -> float:
        """청크 크기 계산 (MB)"""
        chunk_size = 0
        for file_path in chunk:
            try:
                if os.path.exists(file_path):
                    chunk_size += os.path.getsize(file_path)
            except Exception:
                continue
        return chunk_size / (1024 * 1024)
    
    def _calculate_optimal_chunk_size(self, file_paths: List[str]) -> int:
        """최적 청크 크기 계산"""
        if not file_paths:
            return 1
        
        # 샘플 파일들의 평균 크기 계산
        sample_size = min(10, len(file_paths))
        sample_files = file_paths[:sample_size]
        total_sample_size = 0
        
        for file_path in sample_files:
            try:
                if os.path.exists(file_path):
                    total_sample_size += os.path.getsize(file_path)
            except Exception:
                continue
        
        if total_sample_size == 0:
            return 10  # 기본값
        
        avg_file_size_mb = (total_sample_size / sample_size) / (1024 * 1024)
        
        # 메모리 제한을 고려한 청크 크기
        chunk_memory_limit = self.memory_config.chunk_memory_mb
        optimal_chunk_size = max(1, int(chunk_memory_limit / max(avg_file_size_mb, 1.0)))
        
        # 최소 1, 최대 100으로 제한
        return min(max(optimal_chunk_size, 1), 100)
    
    def _force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        self.logger.debug("강제 가비지 컬렉션 실행")
        gc.collect()
        self.stats.gc_count += 1
    
    def cleanup(self):
        """임시 파일 정리"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"임시 디렉토리 정리 완료: {self.temp_dir}")
        except Exception as e:
            self.logger.error(f"임시 디렉토리 정리 실패: {e}")


class StreamingDatasetProcessor(LoggerMixin):
    """스트리밍 데이터셋 처리기"""
    
    def __init__(self, memory_config: MemoryConfig = None,
                 batch_config: BatchProcessingConfig = None,
                 augmentation_config: ParallelAugmentationConfig = None):
        self.memory_config = memory_config or MemoryConfig()
        self.batch_config = batch_config or BatchProcessingConfig(
            chunk_size=16, memory_limit_gb=self.memory_config.max_memory_gb / 2
        )
        self.augmentation_config = augmentation_config or ParallelAugmentationConfig(
            chunk_size=8, memory_limit_gb=self.memory_config.max_memory_gb / 2
        )
        
        self.logger = self.get_logger()
        self.chunked_processor = ChunkedFileProcessor(memory_config)
        self.feature_extractor = OptimizedFeatureExtractor()
        self.augmentor = OptimizedAudioAugmentor()
    
    def process_large_dataset_streaming(self, 
                                      audio_files: List[str],
                                      noise_files: List[str] = None,
                                      output_dir: str = None,
                                      include_features: bool = True,
                                      include_augmentation: bool = True) -> Iterator[Dict]:
        """
        대용량 데이터셋을 스트리밍 방식으로 처리
        
        Args:
            audio_files: 원본 오디오 파일 목록
            noise_files: 노이즈 파일 목록 (옵션)
            output_dir: 출력 디렉토리 (옵션)
            include_features: 특징 추출 포함 여부
            include_augmentation: 데이터 증강 포함 여부
            
        Yields:
            각 배치의 처리 결과
        """
        self.logger.info(f"대용량 스트리밍 처리 시작: {len(audio_files)}개 오디오 파일")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 데이터 증강이 필요한 경우
            if include_augmentation and noise_files:
                self.logger.info("데이터 증강 포함 처리")
                yield from self._process_with_augmentation_streaming(
                    audio_files, noise_files, output_dir, include_features)
            else:
                self.logger.info("특징 추출만 처리")
                yield from self._process_features_only_streaming(
                    audio_files, include_features)
                
        finally:
            # 정리
            self.chunked_processor.cleanup()
    
    def _process_with_augmentation_streaming(self, 
                                           audio_files: List[str],
                                           noise_files: List[str],
                                           output_dir: str,
                                           include_features: bool) -> Iterator[Dict]:
        """증강을 포함한 스트리밍 처리"""
        
        def augmentation_processor(audio_chunk: List[str]) -> Dict:
            """청크별 증강 처리 함수"""
            augmented_files = []
            errors = []
            
            for audio_file in audio_chunk:
                for noise_file in noise_files[:3]:  # 메모리 절약을 위해 노이즈 3개로 제한
                    for snr_level in self.augmentation_config.snr_levels:
                        try:
                            augmented_path = self.augmentor.augment_with_caching(
                                audio_file, noise_file, snr_level, output_dir)
                            
                            if augmented_path:
                                augmented_files.append(augmented_path)
                        except Exception as e:
                            errors.append((audio_file, str(e)))
            
            result = {
                'augmented_files': augmented_files,
                'original_files': audio_chunk,
                'errors': errors,
                'success_count': len(augmented_files),
                'failure_count': len(errors)
            }
            
            # 특징 추출 포함
            if include_features:
                all_files = audio_chunk + augmented_files
                features_result = self._extract_features_from_files(all_files)
                result['features'] = features_result
            
            return result
        
        # 청크별 처리
        yield from self.chunked_processor.process_files_chunked(
            audio_files, augmentation_processor)
    
    def _process_features_only_streaming(self, 
                                       audio_files: List[str],
                                       include_features: bool) -> Iterator[Dict]:
        """특징 추출만 하는 스트리밍 처리"""
        
        def feature_processor(audio_chunk: List[str]) -> Dict:
            """청크별 특징 추출 함수"""
            if not include_features:
                return {
                    'processed_files': audio_chunk,
                    'success_count': len(audio_chunk),
                    'failure_count': 0
                }
            
            return self._extract_features_from_files(audio_chunk)
        
        # 청크별 처리
        yield from self.chunked_processor.process_files_chunked(
            audio_files, feature_processor)
    
    def _extract_features_from_files(self, file_paths: List[str]) -> Dict:
        """파일 목록에서 특징 추출"""
        features = []
        successful_files = []
        errors = []
        
        for file_path in file_paths:
            try:
                feature_vector = self.feature_extractor.extract_features_optimized(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    successful_files.append(file_path)
                else:
                    errors.append((file_path, "특징 추출 실패"))
            except Exception as e:
                errors.append((file_path, str(e)))
        
        return {
            'features': features,
            'successful_files': successful_files,
            'errors': errors,
            'success_count': len(successful_files),
            'failure_count': len(errors)
        }
    
    def get_processing_stats(self) -> ProcessingStats:
        """처리 통계 반환"""
        return self.chunked_processor.stats


class MemoryMappedFileHandler:
    """메모리 맵 파일 핸들러"""
    
    def __init__(self, file_path: str, mode: str = 'r'):
        self.file_path = file_path
        self.mode = mode
        self._file = None
        self._mmap = None
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        try:
            self._file = open(self.file_path, 'rb' if 'r' in self.mode else 'r+b')
            self._mmap = mmap.mmap(self._file.fileno(), 0, 
                                access=mmap.ACCESS_READ if 'r' in self.mode else mmap.ACCESS_WRITE)
            return self._mmap
        except Exception as e:
            self._cleanup()
            raise e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self._cleanup()
    
    def _cleanup(self):
        """리소스 정리"""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()


# 편의 함수들
def process_large_dataset_memory_efficient(
    audio_files: List[str],
    noise_files: List[str] = None,
    output_dir: str = None,
    memory_limit_gb: float = 2.0,
    include_features: bool = True,
    include_augmentation: bool = True,
    progress_callback: Callable = None) -> Dict:
    """
    메모리 효율적 대용량 데이터셋 처리 편의 함수
    
    Args:
        audio_files: 오디오 파일 목록
        noise_files: 노이즈 파일 목록
        output_dir: 출력 디렉토리
        memory_limit_gb: 메모리 제한 (GB)
        include_features: 특징 추출 포함 여부
        include_augmentation: 데이터 증강 포함 여부
        progress_callback: 진행률 콜백
        
    Returns:
        전체 처리 결과 요약
    """
    memory_config = MemoryConfig(max_memory_gb=memory_limit_gb)
    processor = StreamingDatasetProcessor(memory_config=memory_config)
    
    total_results = {
        'total_processed': 0,
        'total_augmented': 0,
        'total_features': 0,
        'total_errors': 0,
        'processing_time': 0,
        'chunks_processed': 0
    }
    
    start_time = time.time()
    
    try:
        for chunk_result in processor.process_large_dataset_streaming(
            audio_files, noise_files, output_dir, include_features, include_augmentation):
            
            # 결과 누적
            total_results['chunks_processed'] += 1
            total_results['total_processed'] += chunk_result.get('success_count', 0)
            total_results['total_errors'] += chunk_result.get('failure_count', 0)
            
            if 'augmented_files' in chunk_result:
                total_results['total_augmented'] += len(chunk_result['augmented_files'])
            
            if 'features' in chunk_result:
                total_results['total_features'] += len(chunk_result['features'].get('features', []))
            
            # 진행률 콜백
            if progress_callback:
                progress = total_results['total_processed'] / len(audio_files)
                progress_callback(progress)
    
    finally:
        total_results['processing_time'] = time.time() - start_time
        stats = processor.get_processing_stats()
        total_results['memory_stats'] = {
            'peak_memory_mb': stats.peak_memory_mb,
            'avg_memory_mb': stats.avg_memory_mb,
            'gc_count': stats.gc_count
        }
    
    return total_results


# 사용 예제
if __name__ == "__main__":
    # 테스트용 간단한 예제
    import glob
    
    # 테스트 파일 찾기
    test_audio_files = glob.glob("data/raw/train/*/*.wav")
    test_noise_files = glob.glob("data/noise/**/*.wav", recursive=True)
    
    if test_audio_files:
        print(f"대용량 데이터셋 테스트: {len(test_audio_files)}개 오디오 파일")
        
        # 진행률 콜백
        def progress_callback(progress):
            print(f"진행률: {progress:.1%}")
        
        # 메모리 효율적 처리
        result = process_large_dataset_memory_efficient(
            test_audio_files[:50],  # 테스트를 위해 50개로 제한
            test_noise_files[:5],   # 노이즈 5개로 제한
            output_dir="temp_output",
            memory_limit_gb=1.0,    # 1GB 제한
            include_features=True,
            include_augmentation=bool(test_noise_files),
            progress_callback=progress_callback
        )
        
        print(f"\n처리 결과:")
        print(f"  처리된 파일: {result['total_processed']}")
        print(f"  증강된 파일: {result['total_augmented']}")
        print(f"  추출된 특징: {result['total_features']}")
        print(f"  에러: {result['total_errors']}")
        print(f"  처리 시간: {result['processing_time']:.2f}초")
        print(f"  청크 수: {result['chunks_processed']}")
        print(f"  피크 메모리: {result['memory_stats']['peak_memory_mb']:.1f}MB")
        print(f"  가비지 컬렉션: {result['memory_stats']['gc_count']}회")
        
        # 임시 출력 디렉토리 정리
        if os.path.exists("temp_output"):
            shutil.rmtree("temp_output")
            print("  임시 출력 디렉토리 정리 완료")
    else:
        print("테스트할 오디오 파일을 찾을 수 없습니다.")