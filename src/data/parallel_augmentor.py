"""
병렬 데이터 증강 모듈

대용량 오디오 데이터의 효율적인 병렬 증강 처리를 위한 최적화된 시스템
"""

import os
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union, Iterator
from pathlib import Path
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
import time
import psutil
from functools import partial
import tempfile
import shutil
import uuid
from collections import defaultdict

from .augmentation import AudioAugmentor, AugmentationResult
from ..utils.logger import LoggerMixin
from ..utils.performance_monitor import PerformanceMonitor
from config import DEFAULT_CONFIG


@dataclass
class ParallelAugmentationConfig:
    """병렬 증강 설정"""
    max_workers: Optional[int] = None  # None이면 CPU 코어 수
    chunk_size: int = 16  # 한 번에 처리할 파일 수
    memory_limit_gb: float = 6.0  # 메모리 사용 제한 (GB)
    temp_dir: Optional[str] = None  # 임시 디렉토리 (None이면 시스템 기본값)
    use_multiprocessing: bool = True  # True: 프로세스, False: 스레드
    cleanup_temp_files: bool = True  # 임시 파일 자동 정리
    snr_levels: List[float] = field(default_factory=lambda: [0, 5, 10])
    augmentation_factor: int = 2  # 증강 배수
    progress_callback: Optional[callable] = None
    error_tolerance: float = 0.1  # 허용 가능한 실패율 (0.0-1.0)


@dataclass 
class ParallelAugmentationResult:
    """병렬 증강 결과"""
    total_processed: int
    total_created: int
    success_count: int
    failure_count: int
    processing_time: float
    memory_peak_mb: float
    augmented_files: List[str]
    errors: List[Tuple[str, str]]  # (파일 경로, 에러 메시지)
    temp_dir_usage_mb: float
    cleanup_success: bool


class OptimizedAudioAugmentor(AudioAugmentor):
    """최적화된 오디오 증강기"""
    
    def __init__(self, config=None):
        super().__init__(config or DEFAULT_CONFIG)
        
        # 메모리 효율을 위한 설정
        self._audio_cache = {}  # 작은 캐시
        self._max_cache_size = 50  # 최대 캐시 항목 수
    
    def augment_with_caching(self, clean_file: str, noise_file: str, 
                           snr_level: float, output_dir: str) -> Optional[str]:
        """
        캐싱을 사용한 최적화된 증강
        
        Args:
            clean_file: 깨끗한 오디오 파일
            noise_file: 노이즈 파일
            snr_level: SNR 레벨
            output_dir: 출력 디렉토리
            
        Returns:
            증강된 파일 경로 또는 None
        """
        try:
            # 오디오 로딩 (캐싱 사용)
            clean_audio = self._load_audio_cached(clean_file)
            noise_audio = self._load_audio_cached(noise_file)
            
            if clean_audio is None or noise_audio is None:
                return None
            
            # 길이 조정
            min_length = min(len(clean_audio), len(noise_audio))
            clean_trimmed = clean_audio[:min_length]
            noise_trimmed = noise_audio[:min_length]
            
            # SNR 조정 및 혼합
            scaled_noise = self.scale_noise_for_snr(clean_trimmed, noise_trimmed, snr_level)
            mixed_audio = self.mix_signals(clean_trimmed, scaled_noise)
            
            # 출력 파일명 생성
            clean_name = Path(clean_file).stem
            noise_name = Path(noise_file).stem
            output_filename = f"{clean_name}_noise_{noise_name}_snr{snr_level:+.1f}dB.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # 파일 저장
            sf.write(output_path, mixed_audio, self.sr)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"증강 실패 {clean_file} + {noise_file}: {e}")
            return None
    
    def _load_audio_cached(self, audio_file: str) -> Optional[np.ndarray]:
        """캐싱을 사용한 오디오 로딩"""
        # 캐시 확인
        if audio_file in self._audio_cache:
            return self._audio_cache[audio_file]
        
        try:
            # 메모리 효율적인 로딩
            audio, _ = sf.read(audio_file, dtype=np.float32)
            
            # 모노로 변환 (필요시)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 캐시에 저장 (크기 제한)
            if len(self._audio_cache) < self._max_cache_size:
                self._audio_cache[audio_file] = audio
            
            return audio
            
        except Exception as e:
            self.logger.error(f"오디오 로딩 실패 {audio_file}: {e}")
            return None
    
    def clear_cache(self):
        """캐시 정리"""
        self._audio_cache.clear()


class ParallelBatchAugmentor(LoggerMixin):
    """병렬 배치 증강기"""
    
    def __init__(self, config: ParallelAugmentationConfig = None):
        self.config = config or ParallelAugmentationConfig()
        self.logger = self.get_logger()
        self.performance_monitor = PerformanceMonitor()
        
        # 임시 디렉토리 설정
        if self.config.temp_dir is None:
            self.config.temp_dir = tempfile.mkdtemp(prefix="wm_augment_")
        else:
            os.makedirs(self.config.temp_dir, exist_ok=True)
        
        # 워커 수 자동 설정
        if self.config.max_workers is None:
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            max_workers_by_memory = max(1, int(memory_gb / self.config.memory_limit_gb * cpu_count))
            self.config.max_workers = min(cpu_count, max_workers_by_memory)
        
        self.logger.info(f"병렬 증강기 초기화: {self.config.max_workers}개 워커, "
                        f"청크 크기: {self.config.chunk_size}")
    
    def augment_directory_parallel(self, audio_dir: str, noise_dir: str, 
                                 output_dir: str) -> ParallelAugmentationResult:
        """
        디렉토리 전체를 병렬로 증강
        
        Args:
            audio_dir: 원본 오디오 디렉토리
            noise_dir: 노이즈 디렉토리  
            output_dir: 출력 디렉토리
            
        Returns:
            병렬 증강 결과
        """
        self.logger.info(f"병렬 디렉토리 증강 시작: {audio_dir}")
        start_time = time.time()
        
        # 성능 모니터링 시작
        self.performance_monitor.start_monitoring()
        
        try:
            # 파일 목록 수집
            audio_files = self._collect_audio_files(audio_dir)
            noise_files = self._collect_audio_files(noise_dir)
            
            if not audio_files:
                raise ValueError(f"오디오 파일이 없습니다: {audio_dir}")
            
            if not noise_files:
                self.logger.warning(f"노이즈 파일이 없습니다: {noise_dir}, 원본 파일만 복사합니다")
                return self._copy_original_files(audio_files, output_dir)
            
            # 증강 작업 생성
            augmentation_tasks = self._create_augmentation_tasks(
                audio_files, noise_files, output_dir)
            
            # 병렬 처리 실행
            if self.config.use_multiprocessing:
                result = self._process_with_multiprocessing(augmentation_tasks)
            else:
                result = self._process_with_threading(augmentation_tasks)
                
        finally:
            # 성능 모니터링 종료
            perf_stats = self.performance_monitor.stop_monitoring()
        
        processing_time = time.time() - start_time
        
        # 임시 디렉토리 사용량 계산
        temp_usage_mb = self._calculate_directory_size(self.config.temp_dir)
        
        # 임시 파일 정리
        cleanup_success = True
        if self.config.cleanup_temp_files:
            cleanup_success = self._cleanup_temp_files()
        
        # 결과 생성
        parallel_result = ParallelAugmentationResult(
            total_processed=len(audio_files),
            total_created=result['total_created'],
            success_count=result['success_count'],
            failure_count=result['failure_count'],
            processing_time=processing_time,
            memory_peak_mb=perf_stats.get('peak_memory_mb', 0),
            augmented_files=result['augmented_files'],
            errors=result['errors'],
            temp_dir_usage_mb=temp_usage_mb,
            cleanup_success=cleanup_success
        )
        
        self.logger.info(f"병렬 증강 완료: {parallel_result.success_count}/{len(audio_files)} 성공, "
                        f"{parallel_result.total_created}개 파일 생성, "
                        f"{processing_time:.2f}초")
        
        return parallel_result
    
    def _collect_audio_files(self, directory: str) -> List[str]:
        """오디오 파일 수집"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac']
        audio_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        return sorted(audio_files)
    
    def _create_augmentation_tasks(self, audio_files: List[str], 
                                 noise_files: List[str], 
                                 output_dir: str) -> List[Dict]:
        """증강 작업 생성"""
        tasks = []
        
        for audio_file in audio_files:
            for noise_file in noise_files:
                for snr_level in self.config.snr_levels:
                    task = {
                        'audio_file': audio_file,
                        'noise_file': noise_file,
                        'snr_level': snr_level,
                        'output_dir': output_dir,
                        'task_id': str(uuid.uuid4())[:8]
                    }
                    tasks.append(task)
        
        self.logger.info(f"증강 작업 생성: {len(tasks)}개 작업")
        return tasks
    
    def _process_with_multiprocessing(self, tasks: List[Dict]) -> Dict:
        """멀티프로세싱을 사용한 병렬 처리"""
        augmented_files = []
        errors = []
        success_count = 0
        failure_count = 0
        
        # 청크로 분할
        chunks = [tasks[i:i + self.config.chunk_size] 
                 for i in range(0, len(tasks), self.config.chunk_size)]
        
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
                    
                    augmented_files.extend(chunk_result['augmented_files'])
                    errors.extend(chunk_result['errors'])
                    success_count += chunk_result['success_count']
                    failure_count += chunk_result['failure_count']
                    
                except Exception as e:
                    self.logger.error(f"청크 처리 실패: {e}")
                    failure_count += len(chunk)
                    errors.extend([(task['audio_file'], str(e)) for task in chunk])
                
                completed_chunks += 1
                
                # 진행률 콜백
                if self.config.progress_callback:
                    progress = completed_chunks / len(chunks)
                    self.config.progress_callback(progress)
        
        return {
            'augmented_files': augmented_files,
            'errors': errors,
            'success_count': success_count,
            'failure_count': failure_count,
            'total_created': len(augmented_files)
        }
    
    def _process_with_threading(self, tasks: List[Dict]) -> Dict:
        """멀티스레딩을 사용한 병렬 처리"""
        augmented_files = []
        errors = []
        success_count = 0
        failure_count = 0
        
        # 공유 증강기 (스레드 안전)
        augmentor = OptimizedAudioAugmentor()
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 개별 작업을 병렬로 처리
            future_to_task = {
                executor.submit(self._process_single_task, task, augmentor): task 
                for task in tasks
            }
            
            completed_tasks = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    if result:
                        augmented_files.append(result)
                        success_count += 1
                    else:
                        failure_count += 1
                        errors.append((task['audio_file'], "증강 실패"))
                        
                except Exception as e:
                    failure_count += 1
                    errors.append((task['audio_file'], str(e)))
                
                completed_tasks += 1
                
                # 진행률 콜백
                if self.config.progress_callback:
                    progress = completed_tasks / len(tasks)
                    self.config.progress_callback(progress)
        
        return {
            'augmented_files': augmented_files,
            'errors': errors,
            'success_count': success_count,
            'failure_count': failure_count,
            'total_created': len(augmented_files)
        }
    
    @staticmethod
    def _process_chunk_worker(task_chunk: List[Dict]) -> Dict:
        """워커 프로세스에서 실행되는 청크 처리 함수"""
        augmentor = OptimizedAudioAugmentor()
        
        augmented_files = []
        errors = []
        success_count = 0
        failure_count = 0
        
        for task in task_chunk:
            try:
                result = augmentor.augment_with_caching(
                    task['audio_file'],
                    task['noise_file'], 
                    task['snr_level'],
                    task['output_dir']
                )
                
                if result:
                    augmented_files.append(result)
                    success_count += 1
                else:
                    failure_count += 1
                    errors.append((task['audio_file'], "증강 실패"))
                    
            except Exception as e:
                failure_count += 1
                errors.append((task['audio_file'], str(e)))
        
        # 캐시 정리
        augmentor.clear_cache()
        
        return {
            'augmented_files': augmented_files,
            'errors': errors,
            'success_count': success_count,
            'failure_count': failure_count
        }
    
    def _process_single_task(self, task: Dict, augmentor: OptimizedAudioAugmentor) -> Optional[str]:
        """단일 작업 처리"""
        return augmentor.augment_with_caching(
            task['audio_file'],
            task['noise_file'],
            task['snr_level'], 
            task['output_dir']
        )
    
    def _copy_original_files(self, audio_files: List[str], output_dir: str) -> ParallelAugmentationResult:
        """노이즈가 없을 때 원본 파일만 복사"""
        self.logger.info("노이즈 파일이 없어 원본 파일 복사")
        
        os.makedirs(output_dir, exist_ok=True)
        copied_files = []
        errors = []
        
        for audio_file in audio_files:
            try:
                filename = os.path.basename(audio_file)
                output_path = os.path.join(output_dir, filename)
                shutil.copy2(audio_file, output_path)
                copied_files.append(output_path)
            except Exception as e:
                errors.append((audio_file, str(e)))
        
        return ParallelAugmentationResult(
            total_processed=len(audio_files),
            total_created=len(copied_files),
            success_count=len(copied_files),
            failure_count=len(errors),
            processing_time=0.1,
            memory_peak_mb=0,
            augmented_files=copied_files,
            errors=errors,
            temp_dir_usage_mb=0,
            cleanup_success=True
        )
    
    def _calculate_directory_size(self, directory: str) -> float:
        """디렉토리 크기 계산 (MB)"""
        total_size = 0
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            self.logger.warning(f"디렉토리 크기 계산 실패: {e}")
        
        return total_size / (1024 * 1024)  # MB로 변환
    
    def _cleanup_temp_files(self) -> bool:
        """임시 파일 정리"""
        try:
            if os.path.exists(self.config.temp_dir):
                shutil.rmtree(self.config.temp_dir)
                self.logger.info(f"임시 디렉토리 정리 완료: {self.config.temp_dir}")
            return True
        except Exception as e:
            self.logger.error(f"임시 파일 정리 실패: {e}")
            return False


class StreamingAugmentor(LoggerMixin):
    """스트리밍 증강기 (메모리 효율성)"""
    
    def __init__(self, config: ParallelAugmentationConfig = None):
        self.config = config or ParallelAugmentationConfig()
        self.logger = self.get_logger()
        self.augmentor = OptimizedAudioAugmentor()
    
    def augment_stream(self, audio_files: List[str], noise_files: List[str],
                      output_dir: str, 
                      result_callback: callable = None) -> Iterator[str]:
        """
        스트리밍 방식으로 증강
        
        Args:
            audio_files: 원본 오디오 파일 목록
            noise_files: 노이즈 파일 목록
            output_dir: 출력 디렉토리
            result_callback: 각 결과를 처리할 콜백
            
        Yields:
            증강된 파일 경로
        """
        self.logger.info(f"스트리밍 증강 시작: {len(audio_files)}개 원본, "
                        f"{len(noise_files)}개 노이즈")
        
        os.makedirs(output_dir, exist_ok=True)
        processed_count = 0
        total_combinations = len(audio_files) * len(noise_files) * len(self.config.snr_levels)
        
        for audio_file in audio_files:
            for noise_file in noise_files:
                for snr_level in self.config.snr_levels:
                    try:
                        augmented_path = self.augmentor.augment_with_caching(
                            audio_file, noise_file, snr_level, output_dir)
                        
                        if augmented_path:
                            if result_callback:
                                result_callback(augmented_path)
                            
                            yield augmented_path
                        
                        processed_count += 1
                        
                        # 진행률 콜백
                        if self.config.progress_callback:
                            progress = processed_count / total_combinations
                            self.config.progress_callback(progress)
                            
                    except Exception as e:
                        self.logger.error(f"스트리밍 증강 실패 {audio_file}: {e}")
                        continue
        
        # 캐시 정리
        self.augmentor.clear_cache()
        self.logger.info(f"스트리밍 증강 완료: {processed_count}개 처리")


# 편의 함수들
def augment_directory_parallel(audio_dir: str, noise_dir: str, output_dir: str,
                              config: ParallelAugmentationConfig = None) -> ParallelAugmentationResult:
    """
    병렬 디렉토리 증강 편의 함수
    
    Args:
        audio_dir: 원본 오디오 디렉토리
        noise_dir: 노이즈 디렉토리
        output_dir: 출력 디렉토리
        config: 병렬 증강 설정
        
    Returns:
        병렬 증강 결과
    """
    augmentor = ParallelBatchAugmentor(config)
    return augmentor.augment_directory_parallel(audio_dir, noise_dir, output_dir)


def augment_streaming(audio_files: List[str], noise_files: List[str], 
                     output_dir: str,
                     result_callback: callable = None,
                     config: ParallelAugmentationConfig = None) -> Iterator[str]:
    """
    스트리밍 증강 편의 함수
    
    Args:
        audio_files: 원본 오디오 파일 목록
        noise_files: 노이즈 파일 목록
        output_dir: 출력 디렉토리
        result_callback: 결과 처리 콜백
        config: 병렬 증강 설정
        
    Yields:
        증강된 파일 경로
    """
    augmentor = StreamingAugmentor(config)
    yield from augmentor.augment_stream(audio_files, noise_files, output_dir, result_callback)


# 사용 예제
if __name__ == "__main__":
    # 간단한 사용 예제
    import tempfile
    
    # 테스트 디렉토리 설정
    audio_dir = "data/raw/train/watermelon_A"
    noise_dir = "data/noise"
    
    if os.path.exists(audio_dir) and os.path.exists(noise_dir):
        with tempfile.TemporaryDirectory() as temp_output:
            print(f"병렬 증강 테스트: {audio_dir} + {noise_dir}")
            
            # 진행률 콜백
            def progress_callback(progress):
                print(f"진행률: {progress:.1%}")
            
            # 병렬 증강 설정
            config = ParallelAugmentationConfig(
                max_workers=4,
                chunk_size=8,
                snr_levels=[0, 5, 10],
                progress_callback=progress_callback,
                use_multiprocessing=True,
                cleanup_temp_files=True
            )
            
            # 병렬 증강 실행
            result = augment_directory_parallel(audio_dir, noise_dir, temp_output, config)
            
            print(f"증강 결과:")
            print(f"  처리된 파일: {result.total_processed}")
            print(f"  생성된 파일: {result.total_created}")
            print(f"  성공/실패: {result.success_count}/{result.failure_count}")
            print(f"  처리 시간: {result.processing_time:.2f}초")
            print(f"  피크 메모리: {result.memory_peak_mb:.1f}MB")
            print(f"  임시 공간 사용: {result.temp_dir_usage_mb:.1f}MB")
            print(f"  정리 성공: {result.cleanup_success}")
            
            if result.errors:
                print(f"  에러 {len(result.errors)}개:")
                for file_path, error in result.errors[:3]:
                    print(f"    {file_path}: {error}")
    else:
        print("테스트 디렉토리를 찾을 수 없습니다.")