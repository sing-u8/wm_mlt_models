"""
하드웨어별 최적화 설정 모듈

다양한 하드웨어 환경에서 최적 성능을 위한 자동 설정 시스템
"""

import os
import sys
import platform
import psutil
import multiprocessing as mp
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import logging

from ..utils.logger import LoggerMixin


@dataclass
class HardwareProfile:
    """하드웨어 프로필"""
    profile_name: str
    cpu_cores: int
    memory_gb: float
    storage_type: str  # 'SSD', 'HDD', 'NVMe'
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    platform_type: str = "unknown"  # 'desktop', 'laptop', 'server', 'mobile'
    
    # 최적화 설정
    max_workers: int = field(init=False)
    chunk_size: int = field(init=False)
    memory_limit_gb: float = field(init=False)
    use_multiprocessing: bool = field(init=False)
    prefetch_count: int = field(init=False)
    
    def __post_init__(self):
        """하드웨어 기반 최적화 설정 자동 계산"""
        self._calculate_optimal_settings()
    
    def _calculate_optimal_settings(self):
        """하드웨어 특성에 맞는 최적 설정 계산"""
        # 워커 수: CPU 코어 기반, 메모리 제약 고려
        memory_workers = max(1, int(self.memory_gb / 2))  # 2GB당 1워커
        self.max_workers = min(self.cpu_cores, memory_workers, 8)  # 최대 8개로 제한
        
        # 청크 크기: 메모리와 스토리지 타입 고려
        base_chunk_size = 16
        if self.storage_type == 'HDD':
            self.chunk_size = max(8, base_chunk_size // 2)  # HDD는 작은 청크
        elif self.storage_type == 'NVMe':
            self.chunk_size = min(64, base_chunk_size * 2)  # NVMe는 큰 청크
        else:  # SSD
            self.chunk_size = base_chunk_size
        
        # 메모리 제한: 전체 메모리의 70%
        self.memory_limit_gb = self.memory_gb * 0.7
        
        # 멀티프로세싱 vs 멀티스레딩
        # CPU 집약적 작업에는 멀티프로세싱이 유리
        self.use_multiprocessing = self.cpu_cores >= 4
        
        # 프리페치 수: 메모리 기반
        if self.memory_gb >= 8:
            self.prefetch_count = 3
        elif self.memory_gb >= 4:
            self.prefetch_count = 2
        else:
            self.prefetch_count = 1


@dataclass
class PerformancePreset:
    """성능 프리셋"""
    name: str
    description: str
    memory_usage: str  # 'low', 'medium', 'high'
    processing_speed: str  # 'slow', 'medium', 'fast'
    quality_level: str  # 'basic', 'standard', 'high'
    
    # 특징 추출 설정
    feature_extraction_workers: int
    feature_chunk_size: int
    
    # 데이터 증강 설정  
    augmentation_workers: int
    augmentation_chunk_size: int
    snr_levels: List[float]
    
    # 메모리 관리
    max_memory_gb: float
    gc_threshold: float
    enable_caching: bool


class HardwareDetector(LoggerMixin):
    """하드웨어 자동 감지"""
    
    def __init__(self):
        self.logger = self.get_logger()
    
    def detect_hardware(self) -> HardwareProfile:
        """현재 시스템의 하드웨어 프로필 감지"""
        try:
            # CPU 정보
            cpu_cores = mp.cpu_count()
            
            # 메모리 정보 (GB)
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            
            # 스토리지 타입 감지
            storage_type = self._detect_storage_type()
            
            # GPU 감지
            gpu_available, gpu_memory_gb = self._detect_gpu()
            
            # 플랫폼 타입 추정
            platform_type = self._detect_platform_type()
            
            profile = HardwareProfile(
                profile_name=f"auto_detected_{platform.node()}",
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                storage_type=storage_type,
                gpu_available=gpu_available,
                gpu_memory_gb=gpu_memory_gb,
                platform_type=platform_type
            )
            
            self.logger.info(f"하드웨어 감지 완료: {cpu_cores}코어, {memory_gb:.1f}GB RAM, {storage_type}")
            return profile
            
        except Exception as e:
            self.logger.error(f"하드웨어 감지 실패: {e}")
            # 기본 프로필 반환
            return self._get_default_profile()
    
    def _detect_storage_type(self) -> str:
        """스토리지 타입 감지"""
        try:
            system = platform.system().lower()
            
            if system == 'linux':
                return self._detect_linux_storage()
            elif system == 'darwin':  # macOS
                return self._detect_macos_storage()
            elif system == 'windows':
                return self._detect_windows_storage()
            else:
                return 'SSD'  # 기본값
                
        except Exception as e:
            self.logger.warning(f"스토리지 타입 감지 실패: {e}")
            return 'SSD'
    
    def _detect_linux_storage(self) -> str:
        """Linux 스토리지 타입 감지"""
        try:
            # /sys/block에서 블록 디바이스 정보 확인
            for disk in psutil.disk_partitions():
                if disk.mountpoint == '/':
                    device = disk.device.replace('/dev/', '').rstrip('0123456789')
                    rotational_path = f"/sys/block/{device}/queue/rotational"
                    
                    if os.path.exists(rotational_path):
                        with open(rotational_path, 'r') as f:
                            if f.read().strip() == '0':
                                # NVMe 체크
                                if 'nvme' in device:
                                    return 'NVMe'
                                return 'SSD'
                            else:
                                return 'HDD'
            return 'SSD'
        except Exception:
            return 'SSD'
    
    def _detect_macos_storage(self) -> str:
        """macOS 스토리지 타입 감지"""
        try:
            # diskutil 명령 사용
            result = subprocess.run(['diskutil', 'info', '/'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'solid state' in output or 'ssd' in output:
                    if 'nvme' in output or 'pcie' in output:
                        return 'NVMe'
                    return 'SSD'
                elif 'rotational' in output or 'hdd' in output:
                    return 'HDD'
            return 'SSD'  # macOS는 대부분 SSD
        except Exception:
            return 'SSD'
    
    def _detect_windows_storage(self) -> str:
        """Windows 스토리지 타입 감지"""
        try:
            # WMI를 사용한 감지 (선택적)
            import wmi
            c = wmi.WMI()
            for disk in c.Win32_DiskDrive():
                if disk.MediaType:
                    media_type = disk.MediaType.lower()
                    if 'ssd' in media_type or 'solid' in media_type:
                        return 'SSD'
                    elif 'nvme' in media_type:
                        return 'NVMe'
            return 'SSD'
        except ImportError:
            # wmi 모듈이 없으면 기본값
            return 'SSD'
        except Exception:
            return 'SSD'
    
    def _detect_gpu(self) -> Tuple[bool, float]:
        """GPU 감지"""
        try:
            # NVIDIA GPU 감지 시도
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    total_memory = sum(gpu.memoryTotal for gpu in gpus) / 1024  # GB
                    return True, total_memory
            except ImportError:
                pass
            
            # 시스템 명령으로 GPU 감지 시도
            system = platform.system().lower()
            if system == 'linux':
                return self._detect_linux_gpu()
            elif system == 'darwin':
                return self._detect_macos_gpu()
            
            return False, 0.0
            
        except Exception as e:
            self.logger.warning(f"GPU 감지 실패: {e}")
            return False, 0.0
    
    def _detect_linux_gpu(self) -> Tuple[bool, float]:
        """Linux GPU 감지"""
        try:
            # nvidia-smi 명령 시도
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memory_mb = sum(int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip())
                return True, memory_mb / 1024  # GB로 변환
            return False, 0.0
        except Exception:
            return False, 0.0
    
    def _detect_macos_gpu(self) -> Tuple[bool, float]:
        """macOS GPU 감지"""
        try:
            # system_profiler 명령 사용
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # 간단한 GPU 존재 확인 (메모리는 정확하지 않을 수 있음)
                if 'gpu' in result.stdout.lower() or 'graphics' in result.stdout.lower():
                    return True, 4.0  # 추정값
            return False, 0.0
        except Exception:
            return False, 0.0
    
    def _detect_platform_type(self) -> str:
        """플랫폼 타입 감지"""
        try:
            # 배터리 존재 여부로 노트북/데스크톱 구분
            try:
                battery = psutil.sensors_battery()
                if battery is not None:
                    return 'laptop'
            except Exception:
                pass
            
            # CPU 코어 수와 메모리로 서버/데스크톱 구분
            cpu_cores = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if cpu_cores >= 16 and memory_gb >= 32:
                return 'server'
            elif cpu_cores >= 4 and memory_gb >= 8:
                return 'desktop'
            else:
                return 'mobile'
                
        except Exception:
            return 'desktop'
    
    def _get_default_profile(self) -> HardwareProfile:
        """기본 하드웨어 프로필"""
        return HardwareProfile(
            profile_name="default",
            cpu_cores=4,
            memory_gb=8.0,
            storage_type='SSD',
            gpu_available=False,
            gpu_memory_gb=0.0,
            platform_type='desktop'
        )


class PerformancePresetManager:
    """성능 프리셋 관리자"""
    
    def __init__(self):
        self.presets = self._initialize_presets()
    
    def _initialize_presets(self) -> Dict[str, PerformancePreset]:
        """기본 성능 프리셋 초기화"""
        return {
            'low_memory': PerformancePreset(
                name='low_memory',
                description='메모리 사용량 최소화 (2GB 이하 시스템)',
                memory_usage='low',
                processing_speed='slow',
                quality_level='basic',
                feature_extraction_workers=1,
                feature_chunk_size=8,
                augmentation_workers=1,
                augmentation_chunk_size=4,
                snr_levels=[0, 10],
                max_memory_gb=1.5,
                gc_threshold=0.6,
                enable_caching=False
            ),
            
            'balanced': PerformancePreset(
                name='balanced',
                description='균형잡힌 성능과 메모리 사용 (4-8GB 시스템)',
                memory_usage='medium',
                processing_speed='medium',
                quality_level='standard',
                feature_extraction_workers=2,
                feature_chunk_size=16,
                augmentation_workers=2,
                augmentation_chunk_size=8,
                snr_levels=[0, 5, 10],
                max_memory_gb=4.0,
                gc_threshold=0.75,
                enable_caching=True
            ),
            
            'high_performance': PerformancePreset(
                name='high_performance',
                description='최대 성능 (8GB 이상 시스템)',
                memory_usage='high',
                processing_speed='fast',
                quality_level='high',
                feature_extraction_workers=4,
                feature_chunk_size=32,
                augmentation_workers=4,
                augmentation_chunk_size=16,
                snr_levels=[0, 5, 10, 15],
                max_memory_gb=6.0,
                gc_threshold=0.85,
                enable_caching=True
            ),
            
            'server': PerformancePreset(
                name='server',
                description='서버 환경 최적화 (16GB 이상)',
                memory_usage='high',
                processing_speed='fast',
                quality_level='high',
                feature_extraction_workers=8,
                feature_chunk_size=64,
                augmentation_workers=6,
                augmentation_chunk_size=32,
                snr_levels=[0, 5, 10, 15, 20],
                max_memory_gb=12.0,
                gc_threshold=0.9,
                enable_caching=True
            )
        }
    
    def get_recommended_preset(self, hardware_profile: HardwareProfile) -> PerformancePreset:
        """하드웨어 프로필에 맞는 추천 프리셋"""
        memory_gb = hardware_profile.memory_gb
        
        if memory_gb < 3:
            return self.presets['low_memory']
        elif memory_gb < 8:
            return self.presets['balanced']
        elif memory_gb < 16:
            return self.presets['high_performance']
        else:
            return self.presets['server']
    
    def get_preset(self, preset_name: str) -> Optional[PerformancePreset]:
        """특정 프리셋 반환"""
        return self.presets.get(preset_name)
    
    def list_presets(self) -> List[str]:
        """사용 가능한 프리셋 목록"""
        return list(self.presets.keys())


class HardwareConfigManager(LoggerMixin):
    """하드웨어 설정 관리자"""
    
    def __init__(self, config_dir: str = None):
        self.logger = self.get_logger()
        self.config_dir = Path(config_dir or "config/hardware")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.detector = HardwareDetector()
        self.preset_manager = PerformancePresetManager()
        
        # 현재 설정
        self.current_profile: Optional[HardwareProfile] = None
        self.current_preset: Optional[PerformancePreset] = None
    
    def auto_configure(self) -> Tuple[HardwareProfile, PerformancePreset]:
        """자동 하드웨어 감지 및 설정"""
        self.logger.info("하드웨어 자동 감지 및 설정 시작")
        
        # 하드웨어 감지
        self.current_profile = self.detector.detect_hardware()
        
        # 추천 프리셋 선택
        self.current_preset = self.preset_manager.get_recommended_preset(self.current_profile)
        
        # 설정 저장
        self._save_current_config()
        
        self.logger.info(f"자동 설정 완료: {self.current_preset.name} 프리셋")
        return self.current_profile, self.current_preset
    
    def set_preset(self, preset_name: str) -> bool:
        """수동 프리셋 설정"""
        preset = self.preset_manager.get_preset(preset_name)
        if preset:
            self.current_preset = preset
            self._save_current_config()
            self.logger.info(f"프리셋 변경: {preset_name}")
            return True
        else:
            self.logger.error(f"존재하지 않는 프리셋: {preset_name}")
            return False
    
    def get_current_config(self) -> Dict:
        """현재 설정 반환"""
        if not self.current_profile or not self.current_preset:
            self.auto_configure()
        
        return {
            'hardware_profile': {
                'profile_name': self.current_profile.profile_name,
                'cpu_cores': self.current_profile.cpu_cores,
                'memory_gb': self.current_profile.memory_gb,
                'storage_type': self.current_profile.storage_type,
                'platform_type': self.current_profile.platform_type,
                'max_workers': self.current_profile.max_workers,
                'chunk_size': self.current_profile.chunk_size,
                'memory_limit_gb': self.current_profile.memory_limit_gb,
                'use_multiprocessing': self.current_profile.use_multiprocessing
            },
            'performance_preset': {
                'name': self.current_preset.name,
                'description': self.current_preset.description,
                'feature_extraction_workers': self.current_preset.feature_extraction_workers,
                'feature_chunk_size': self.current_preset.feature_chunk_size,
                'augmentation_workers': self.current_preset.augmentation_workers,
                'augmentation_chunk_size': self.current_preset.augmentation_chunk_size,
                'snr_levels': self.current_preset.snr_levels,
                'max_memory_gb': self.current_preset.max_memory_gb,
                'enable_caching': self.current_preset.enable_caching
            }
        }
    
    def _save_current_config(self):
        """현재 설정 저장"""
        try:
            config_file = self.config_dir / "current_config.json"
            config = self.get_current_config()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"설정 저장 완료: {config_file}")
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {e}")
    
    def load_saved_config(self) -> bool:
        """저장된 설정 로드"""
        try:
            config_file = self.config_dir / "current_config.json"
            if not config_file.exists():
                return False
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 설정 복원 (간단한 버전)
            hw_config = config.get('hardware_profile', {})
            preset_config = config.get('performance_preset', {})
            
            if hw_config and preset_config:
                self.logger.info("저장된 설정 로드 완료")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {e}")
            return False
    
    def print_system_info(self):
        """시스템 정보 출력"""
        if not self.current_profile:
            self.auto_configure()
        
        print("\n=== 시스템 정보 ===")
        print(f"프로필: {self.current_profile.profile_name}")
        print(f"CPU: {self.current_profile.cpu_cores}코어")
        print(f"메모리: {self.current_profile.memory_gb:.1f}GB")
        print(f"스토리지: {self.current_profile.storage_type}")
        print(f"플랫폼: {self.current_profile.platform_type}")
        print(f"GPU: {'있음' if self.current_profile.gpu_available else '없음'}")
        
        print("\n=== 최적화 설정 ===")
        print(f"워커 수: {self.current_profile.max_workers}")
        print(f"청크 크기: {self.current_profile.chunk_size}")
        print(f"메모리 제한: {self.current_profile.memory_limit_gb:.1f}GB")
        print(f"멀티프로세싱: {'사용' if self.current_profile.use_multiprocessing else '미사용'}")
        
        print(f"\n=== 성능 프리셋: {self.current_preset.name} ===")
        print(f"설명: {self.current_preset.description}")
        print(f"특징 추출 워커: {self.current_preset.feature_extraction_workers}")
        print(f"증강 워커: {self.current_preset.augmentation_workers}")
        print(f"SNR 레벨: {self.current_preset.snr_levels}")


# 전역 설정 관리자 인스턴스
_config_manager = None

def get_hardware_config() -> HardwareConfigManager:
    """전역 하드웨어 설정 관리자 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = HardwareConfigManager()
        # 저장된 설정이 있으면 로드, 없으면 자동 감지
        if not _config_manager.load_saved_config():
            _config_manager.auto_configure()
    return _config_manager


# 편의 함수들
def get_optimal_workers() -> int:
    """최적 워커 수 반환"""
    config = get_hardware_config().get_current_config()
    return config['hardware_profile']['max_workers']

def get_optimal_chunk_size() -> int:
    """최적 청크 크기 반환"""
    config = get_hardware_config().get_current_config()
    return config['hardware_profile']['chunk_size']

def get_memory_limit() -> float:
    """메모리 제한 반환 (GB)"""
    config = get_hardware_config().get_current_config()
    return min(
        config['hardware_profile']['memory_limit_gb'],
        config['performance_preset']['max_memory_gb']
    )

def should_use_multiprocessing() -> bool:
    """멀티프로세싱 사용 여부 반환"""
    config = get_hardware_config().get_current_config()
    return config['hardware_profile']['use_multiprocessing']


# 사용 예제
if __name__ == "__main__":
    # 하드웨어 설정 관리자 초기화
    config_manager = HardwareConfigManager()
    
    # 시스템 정보 출력
    config_manager.print_system_info()
    
    # 사용 가능한 프리셋 출력
    print(f"\n=== 사용 가능한 프리셋 ===")
    for preset_name in config_manager.preset_manager.list_presets():
        preset = config_manager.preset_manager.get_preset(preset_name)
        print(f"{preset_name}: {preset.description}")
    
    # 현재 설정 출력
    print(f"\n=== 현재 설정 ===")
    current_config = config_manager.get_current_config()
    print(f"최적 워커 수: {get_optimal_workers()}")
    print(f"최적 청크 크기: {get_optimal_chunk_size()}")
    print(f"메모리 제한: {get_memory_limit():.1f}GB")
    print(f"멀티프로세싱: {should_use_multiprocessing()}")