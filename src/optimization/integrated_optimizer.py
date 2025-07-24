"""
통합 최적화 모듈

모든 최적화 구성요소를 통합하여 하드웨어별 최적 성능을 제공하는 통합 시스템
"""

import os
import sys
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json

# 최적화 구성요소 임포트
from ..audio.batch_processor import BatchFeatureProcessor, BatchProcessingConfig, OptimizedFeatureExtractor
from ..data.parallel_augmentor import ParallelBatchAugmentor, ParallelAugmentationConfig
from ..data.large_dataset_processor import StreamingDatasetProcessor, MemoryConfig, process_large_dataset_memory_efficient
from ..config.hardware_config import HardwareConfigManager, get_hardware_config
from ..utils.logger import LoggerMixin
from ..utils.performance_monitor import PerformanceMonitor


@dataclass
class OptimizationProfile:
    """최적화 프로필"""
    profile_name: str
    hardware_optimized: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    enable_batch_processing: bool = True
    enable_streaming: bool = False
    max_memory_usage_gb: float = 4.0
    target_processing_speed: str = "balanced"  # "fast", "balanced", "memory_efficient"


@dataclass 
class OptimizationResult:
    """최적화 결과"""
    total_files_processed: int
    total_features_extracted: int
    total_augmented_files: int
    processing_time_seconds: float
    peak_memory_mb: float
    average_memory_mb: float
    files_per_second: float
    memory_efficiency_score: float  # 0-1, 높을수록 효율적
    overall_performance_score: float  # 0-1, 높을수록 좋음


class IntegratedOptimizer(LoggerMixin):
    """통합 최적화 처리기"""
    
    def __init__(self, optimization_profile: OptimizationProfile = None):
        self.logger = self.get_logger()
        self.profile = optimization_profile or self._create_auto_profile()
        
        # 하드웨어 설정 관리자
        self.hardware_config = get_hardware_config()
        
        # 성능 모니터
        self.performance_monitor = PerformanceMonitor()
        
        # 최적화 구성요소 초기화
        self._initialize_optimized_components()
        
        self.logger.info(f"통합 최적화기 초기화: {self.profile.profile_name}")
    
    def _create_auto_profile(self) -> OptimizationProfile:
        """하드웨어 기반 자동 프로필 생성"""
        hw_config = get_hardware_config().get_current_config()
        memory_gb = hw_config['hardware_profile']['memory_gb']
        
        if memory_gb < 4:
            return OptimizationProfile(
                profile_name="memory_conserving",
                enable_streaming=True,
                enable_batch_processing=False,
                max_memory_usage_gb=min(2.0, memory_gb * 0.7),
                target_processing_speed="memory_efficient"
            )
        elif memory_gb < 8:
            return OptimizationProfile(
                profile_name="balanced",
                max_memory_usage_gb=min(4.0, memory_gb * 0.7),
                target_processing_speed="balanced"
            )
        else:
            return OptimizationProfile(
                profile_name="high_performance",
                max_memory_usage_gb=min(6.0, memory_gb * 0.7),
                target_processing_speed="fast"
            )
    
    def _initialize_optimized_components(self):
        """최적화된 구성요소 초기화"""
        hw_config = self.hardware_config.get_current_config()
        hw_profile = hw_config['hardware_profile']
        perf_preset = hw_config['performance_preset']
        
        # 배치 처리 설정
        self.batch_config = BatchProcessingConfig(
            max_workers=perf_preset['feature_extraction_workers'],
            chunk_size=perf_preset['feature_chunk_size'],
            memory_limit_gb=min(
                self.profile.max_memory_usage_gb,
                perf_preset['max_memory_gb']
            ),
            use_multiprocessing=hw_profile['use_multiprocessing'],
            cache_features=perf_preset['enable_caching']
        )
        
        # 병렬 증강 설정
        self.augmentation_config = ParallelAugmentationConfig(
            max_workers=perf_preset['augmentation_workers'],
            chunk_size=perf_preset['augmentation_chunk_size'],
            memory_limit_gb=self.profile.max_memory_usage_gb / 2,
            snr_levels=perf_preset['snr_levels'],
            use_multiprocessing=hw_profile['use_multiprocessing']
        )
        
        # 메모리 설정
        self.memory_config = MemoryConfig(
            max_memory_gb=self.profile.max_memory_usage_gb,
            chunk_memory_mb=min(256, self.profile.max_memory_usage_gb * 128),
            gc_threshold=0.8 if self.profile.target_processing_speed == "fast" else 0.6
        )
        
        # 구성요소 초기화
        if self.profile.enable_batch_processing:
            self.batch_processor = BatchFeatureProcessor(self.batch_config)
        
        if self.profile.enable_parallel_processing:
            self.parallel_augmentor = ParallelBatchAugmentor(self.augmentation_config)
        
        if self.profile.enable_streaming or self.profile.enable_memory_optimization:
            self.streaming_processor = StreamingDatasetProcessor(
                memory_config=self.memory_config,
                batch_config=self.batch_config,
                augmentation_config=self.augmentation_config
            )
    
    def process_dataset_optimized(self,
                                audio_files: List[str],
                                noise_files: List[str] = None,
                                output_dir: str = None,
                                extract_features: bool = True,
                                perform_augmentation: bool = True,
                                progress_callback: Callable = None) -> OptimizationResult:
        """
        최적화된 데이터셋 처리
        
        Args:
            audio_files: 처리할 오디오 파일 목록
            noise_files: 노이즈 파일 목록 (증강용)
            output_dir: 출력 디렉토리
            extract_features: 특징 추출 수행 여부
            perform_augmentation: 데이터 증강 수행 여부
            progress_callback: 진행률 콜백 함수
            
        Returns:
            최적화 결과
        """
        self.logger.info(f"최적화된 데이터셋 처리 시작: {len(audio_files)}개 파일")
        start_time = time.time()
        
        # 성능 모니터링 시작
        self.performance_monitor.start_monitoring()
        
        try:
            # 처리 방식 선택
            if self.profile.enable_streaming or len(audio_files) > 1000:
                return self._process_streaming(
                    audio_files, noise_files, output_dir, 
                    extract_features, perform_augmentation, progress_callback)
            else:
                return self._process_batch(
                    audio_files, noise_files, output_dir,
                    extract_features, perform_augmentation, progress_callback)
                
        finally:
            # 성능 모니터링 종료
            perf_stats = self.performance_monitor.stop_monitoring()
            processing_time = time.time() - start_time
            
            self.logger.info(f"데이터셋 처리 완료: {processing_time:.2f}초, "
                           f"피크 메모리: {perf_stats.get('peak_memory_mb', 0):.1f}MB")
    
    def _process_streaming(self,
                         audio_files: List[str],
                         noise_files: List[str],
                         output_dir: str,
                         extract_features: bool,
                         perform_augmentation: bool,
                         progress_callback: Callable) -> OptimizationResult:
        """스트리밍 방식 처리"""
        self.logger.info("스트리밍 방식으로 처리")
        
        result = process_large_dataset_memory_efficient(
            audio_files=audio_files,
            noise_files=noise_files,
            output_dir=output_dir,
            memory_limit_gb=self.profile.max_memory_usage_gb,
            include_features=extract_features,
            include_augmentation=perform_augmentation,
            progress_callback=progress_callback
        )
        
        return self._create_optimization_result(result, len(audio_files))
    
    def _process_batch(self,
                     audio_files: List[str],
                     noise_files: List[str],
                     output_dir: str,
                     extract_features: bool,
                     perform_augmentation: bool,
                     progress_callback: Callable) -> OptimizationResult:
        """배치 방식 처리"""
        self.logger.info("배치 방식으로 처리")
        
        total_processed = 0
        total_features = 0
        total_augmented = 0 
        all_files = audio_files.copy()
        
        # 데이터 증강 먼저 수행
        if perform_augmentation and noise_files and self.profile.enable_parallel_processing:
            self.logger.info("병렬 데이터 증강 수행")
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                aug_result = self.parallel_augmentor.augment_directory_parallel(
                    os.path.dirname(audio_files[0]) if audio_files else "",
                    os.path.dirname(noise_files[0]) if noise_files else "",
                    output_dir
                )
                
                total_augmented = aug_result.total_created
                all_files.extend(aug_result.augmented_files)
        
        # 특징 추출 수행
        if extract_features and self.profile.enable_batch_processing:
            self.logger.info("배치 특징 추출 수행")
            
            # 진행률 콜백 설정
            if progress_callback:
                self.batch_config.progress_callback = progress_callback
            
            feature_result = self.batch_processor.process_files(all_files)
            total_features = feature_result.success_count
            total_processed = len(all_files)
        
        # 결과 생성
        mock_result = {
            'total_processed': total_processed,
            'total_augmented': total_augmented,
            'total_features': total_features,
            'processing_time': time.time() - time.time(),  # 임시
        }
        
        return self._create_optimization_result(mock_result, len(audio_files))
    
    def _create_optimization_result(self, processing_result: Dict, original_file_count: int) -> OptimizationResult:
        """최적화 결과 생성"""
        processing_time = processing_result.get('processing_time', 1.0)
        total_processed = processing_result.get('total_processed', original_file_count)
        
        # 성능 점수 계산
        files_per_second = max(0.1, total_processed / processing_time)
        
        # 메모리 효율성 점수 (낮은 메모리 사용량일수록 높은 점수)
        peak_memory = processing_result.get('memory_stats', {}).get('peak_memory_mb', 1000)
        memory_limit_mb = self.profile.max_memory_usage_gb * 1024
        memory_efficiency = max(0.0, 1.0 - (peak_memory / memory_limit_mb))
        
        # 전체 성능 점수 (속도 + 메모리 효율성)
        speed_score = min(1.0, files_per_second / 10.0)  # 10 files/sec를 최대로 정규화
        overall_performance = (speed_score * 0.6) + (memory_efficiency * 0.4)
        
        return OptimizationResult(
            total_files_processed=total_processed,
            total_features_extracted=processing_result.get('total_features', 0),
            total_augmented_files=processing_result.get('total_augmented', 0),
            processing_time_seconds=processing_time,
            peak_memory_mb=peak_memory,
            average_memory_mb=processing_result.get('memory_stats', {}).get('avg_memory_mb', peak_memory * 0.7),
            files_per_second=files_per_second,
            memory_efficiency_score=memory_efficiency,
            overall_performance_score=overall_performance
        )
    
    def benchmark_system(self, test_files: List[str] = None, num_test_files: int = 20) -> Dict:
        """
        시스템 성능 벤치마크
        
        Args:
            test_files: 테스트용 파일 목록 (None이면 임시 파일 생성)
            num_test_files: 테스트 파일 수
            
        Returns:
            벤치마크 결과
        """
        self.logger.info("시스템 성능 벤치마크 시작")
        
        # 테스트 파일 준비
        if test_files is None:
            test_files = self._generate_test_files(num_test_files)
        else:
            test_files = test_files[:num_test_files]
        
        if not test_files:
            self.logger.error("벤치마크용 테스트 파일이 없습니다")
            return {}
        
        benchmark_results = {}
        
        # 1. 특징 추출만 벤치마크
        self.logger.info("특징 추출 벤치마크")
        feature_result = self.process_dataset_optimized(
            audio_files=test_files,
            extract_features=True,
            perform_augmentation=False
        )
        benchmark_results['feature_extraction'] = {
            'files_per_second': feature_result.files_per_second,
            'memory_efficiency': feature_result.memory_efficiency_score,
            'peak_memory_mb': feature_result.peak_memory_mb
        }
        
        # 2. 하드웨어 정보
        hw_config = self.hardware_config.get_current_config()
        benchmark_results['system_info'] = {
            'cpu_cores': hw_config['hardware_profile']['cpu_cores'],
            'memory_gb': hw_config['hardware_profile']['memory_gb'],
            'storage_type': hw_config['hardware_profile']['storage_type'],
            'platform_type': hw_config['hardware_profile']['platform_type'],
            'current_preset': hw_config['performance_preset']['name']
        }
        
        # 3. 권장 설정
        benchmark_results['recommendations'] = self._generate_recommendations(benchmark_results)
        
        self.logger.info("벤치마크 완료")
        return benchmark_results
    
    def _generate_test_files(self, num_files: int) -> List[str]:
        """테스트용 파일 목록 생성"""
        import glob
        
        # 실제 오디오 파일 찾기
        patterns = [
            "data/raw/train/*/*.wav",
            "data/raw/*/*.wav", 
            "**/*.wav"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            if files:
                return files[:num_files]
        
        self.logger.warning("테스트용 오디오 파일을 찾을 수 없습니다")
        return []
    
    def _generate_recommendations(self, benchmark_results: Dict) -> Dict:
        """성능 결과 기반 권장사항 생성"""
        recommendations = {}
        
        feature_result = benchmark_results.get('feature_extraction', {})
        files_per_sec = feature_result.get('files_per_second', 0)
        memory_efficiency = feature_result.get('memory_efficiency', 0)
        
        # 성능 기반 권장사항
        if files_per_sec < 2.0:
            recommendations['performance'] = [
                "CPU 코어 수가 부족할 수 있습니다",
                "고성능 프리셋 사용을 고려하세요",
                "스토리지 성능을 확인하세요"
            ]
        elif files_per_sec > 10.0:
            recommendations['performance'] = [
                "뛰어난 성능입니다",
                "더 큰 배치 크기 사용을 고려하세요"
            ]
        else:
            recommendations['performance'] = ["성능이 양호합니다"]
        
        # 메모리 기반 권장사항
        if memory_efficiency < 0.5:
            recommendations['memory'] = [
                "메모리 사용량이 높습니다",
                "스트리밍 모드 사용을 권장합니다",
                "배치 크기를 줄여보세요"
            ]
        else:
            recommendations['memory'] = ["메모리 사용량이 적절합니다"]
        
        return recommendations
    
    def optimize_for_dataset_size(self, estimated_dataset_size: int) -> 'IntegratedOptimizer':
        """데이터셋 크기에 맞는 최적화"""
        if estimated_dataset_size > 10000:
            # 대용량 데이터셋 - 스트리밍 모드
            self.profile.enable_streaming = True
            self.profile.enable_batch_processing = False
            self.profile.target_processing_speed = "memory_efficient"
            
        elif estimated_dataset_size > 1000:
            # 중간 크기 - 배치 처리
            self.profile.enable_batch_processing = True
            self.profile.enable_streaming = False
            self.profile.target_processing_speed = "balanced"
            
        else:
            # 소규모 - 고성능 처리
            self.profile.enable_batch_processing = True
            self.profile.enable_streaming = False
            self.profile.target_processing_speed = "fast"
        
        # 구성요소 재초기화
        self._initialize_optimized_components()
        
        self.logger.info(f"데이터셋 크기 {estimated_dataset_size}에 맞게 최적화 완료")
        return self
    
    def save_optimization_report(self, result: OptimizationResult, filepath: str):
        """최적화 결과 보고서 저장"""
        report = {
            'optimization_profile': {
                'profile_name': self.profile.profile_name,
                'hardware_optimized': self.profile.hardware_optimized,
                'target_processing_speed': self.profile.target_processing_speed,
                'max_memory_usage_gb': self.profile.max_memory_usage_gb
            },
            'hardware_configuration': self.hardware_config.get_current_config(),
            'processing_results': {
                'total_files_processed': result.total_files_processed,
                'total_features_extracted': result.total_features_extracted,
                'total_augmented_files': result.total_augmented_files,
                'processing_time_seconds': result.processing_time_seconds,
                'files_per_second': result.files_per_second,
                'peak_memory_mb': result.peak_memory_mb,
                'memory_efficiency_score': result.memory_efficiency_score,
                'overall_performance_score': result.overall_performance_score
            },
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"최적화 보고서 저장: {filepath}")
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")


# 편의 함수들
def create_auto_optimizer() -> IntegratedOptimizer:
    """자동 설정 최적화기 생성"""
    return IntegratedOptimizer()

def create_memory_efficient_optimizer(max_memory_gb: float = 2.0) -> IntegratedOptimizer:
    """메모리 효율 최적화기 생성"""
    profile = OptimizationProfile(
        profile_name="memory_efficient",
        enable_streaming=True,
        enable_batch_processing=False,
        max_memory_usage_gb=max_memory_gb,
        target_processing_speed="memory_efficient"
    )
    return IntegratedOptimizer(profile)

def create_high_performance_optimizer(max_memory_gb: float = 8.0) -> IntegratedOptimizer:
    """고성능 최적화기 생성"""
    profile = OptimizationProfile(
        profile_name="high_performance",
        enable_batch_processing=True,
        enable_parallel_processing=True,
        max_memory_usage_gb=max_memory_gb,
        target_processing_speed="fast"
    )
    return IntegratedOptimizer(profile)


# 사용 예제
if __name__ == "__main__":
    # 자동 최적화기 생성 및 테스트
    optimizer = create_auto_optimizer()
    
    # 시스템 벤치마크
    print("시스템 벤치마크 실행 중...")
    benchmark = optimizer.benchmark_system()
    
    if benchmark:
        print("\n=== 벤치마크 결과 ===")
        print(f"특징 추출 속도: {benchmark['feature_extraction']['files_per_second']:.1f} files/sec")
        print(f"메모리 효율성: {benchmark['feature_extraction']['memory_efficiency']:.1%}")
        print(f"피크 메모리: {benchmark['feature_extraction']['peak_memory_mb']:.1f}MB")
        
        print("\n=== 시스템 정보 ===")
        sys_info = benchmark['system_info']
        print(f"CPU: {sys_info['cpu_cores']}코어")
        print(f"메모리: {sys_info['memory_gb']:.1f}GB")
        print(f"스토리지: {sys_info['storage_type']}")
        print(f"현재 프리셋: {sys_info['current_preset']}")
        
        print("\n=== 권장사항 ===")
        for category, recommendations in benchmark['recommendations'].items():
            print(f"{category}:")
            for rec in recommendations:
                print(f"  - {rec}")
    else:
        print("벤치마크 실행 실패 - 테스트 파일이 없습니다")