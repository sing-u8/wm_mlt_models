"""
성능 벤치마킹 및 메모리 모니터링 모듈

파이프라인의 성능을 측정하고 리소스 사용량을 모니터링합니다.
"""

import time
import json
import gc
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..utils.logger import LoggerMixin


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    step_name: str
    start_time: float
    end_time: float
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    memory_used: float
    cpu_percent: float
    system_info: Dict[str, Any]
    custom_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass
class SystemResourceInfo:
    """시스템 리소스 정보"""
    cpu_count: int
    memory_total: float  # GB
    memory_available: float  # GB
    disk_total: float  # GB
    disk_free: float  # GB
    python_version: str
    platform: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


class PerformanceMonitor(LoggerMixin):
    """
    성능 모니터링 클래스
    
    실행 시간, 메모리 사용량, CPU 사용률 등을 측정합니다.
    """
    
    def __init__(self, enable_monitoring: bool = True):
        """
        성능 모니터를 초기화합니다.
        
        Parameters:
        -----------
        enable_monitoring : bool
            모니터링 활성화 여부
        """
        self.enable_monitoring = enable_monitoring
        self.metrics_history = []
        self.current_metrics = {}
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        if not PSUTIL_AVAILABLE and enable_monitoring:
            self.logger.warning("psutil이 설치되지 않아 제한된 성능 모니터링만 가능합니다.")
            self.logger.warning("전체 기능을 위해 'pip install psutil' 실행을 권장합니다.")
    
    def get_system_info(self) -> SystemResourceInfo:
        """
        현재 시스템 정보를 수집합니다.
        
        Returns:
        --------
        SystemResourceInfo
            시스템 리소스 정보
        """
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            return SystemResourceInfo(
                cpu_count=psutil.cpu_count(),
                memory_total=memory_info.total / (1024**3),  # GB
                memory_available=memory_info.available / (1024**3),  # GB
                disk_total=disk_info.total / (1024**3),  # GB
                disk_free=disk_info.free / (1024**3),  # GB
                python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                platform=__import__('platform').system()
            )
        else:
            # psutil 없이 기본 정보만
            import platform
            import sys
            
            return SystemResourceInfo(
                cpu_count=1,  # 알 수 없음
                memory_total=0.0,
                memory_available=0.0,
                disk_total=0.0,
                disk_free=0.0,
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                platform=platform.system()
            )
    
    def get_current_memory_usage(self) -> float:
        """
        현재 메모리 사용량을 MB 단위로 반환합니다.
        
        Returns:
        --------
        float
            메모리 사용량 (MB)
        """
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)  # MB
            except:
                return 0.0
        else:
            # psutil 없이는 정확한 측정 불가
            return 0.0
    
    def get_current_cpu_usage(self) -> float:
        """
        현재 CPU 사용률을 반환합니다.
        
        Returns:
        --------
        float
            CPU 사용률 (%)
        """
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent(interval=0.1)
            except:
                return 0.0
        else:
            return 0.0
    
    def start_step_monitoring(self, step_name: str) -> Dict[str, Any]:
        """
        단계별 성능 모니터링을 시작합니다.
        
        Parameters:
        -----------
        step_name : str
            모니터링할 단계 이름
            
        Returns:
        --------
        Dict[str, Any]
            시작 시점의 메트릭
        """
        if not self.enable_monitoring:
            return {}
        
        # 가비지 컬렉션으로 메모리 정리
        gc.collect()
        
        start_metrics = {
            'step_name': step_name,
            'start_time': time.time(),
            'memory_before': self.get_current_memory_usage(),
            'cpu_before': self.get_current_cpu_usage(),
            'system_info': self.get_system_info().to_dict()
        }
        
        self.current_metrics[step_name] = start_metrics
        
        self.logger.info(f"🔍 {step_name} 성능 모니터링 시작")
        self.logger.info(f"  시작 메모리: {start_metrics['memory_before']:.2f} MB")
        self.logger.info(f"  시작 CPU: {start_metrics['cpu_before']:.1f}%")
        
        return start_metrics
    
    def end_step_monitoring(self, step_name: str, 
                          custom_metrics: Dict[str, Any] = None) -> PerformanceMetrics:
        """
        단계별 성능 모니터링을 종료하고 결과를 반환합니다.
        
        Parameters:
        -----------
        step_name : str
            모니터링 중인 단계 이름
        custom_metrics : Dict[str, Any], optional
            사용자 정의 메트릭
            
        Returns:
        --------
        PerformanceMetrics
            성능 메트릭 결과
        """
        if not self.enable_monitoring or step_name not in self.current_metrics:
            # 더미 메트릭 반환
            return PerformanceMetrics(
                step_name=step_name,
                start_time=time.time(),
                end_time=time.time(),
                execution_time=0.0,
                memory_before=0.0,
                memory_after=0.0,
                memory_peak=0.0,
                memory_used=0.0,
                cpu_percent=0.0,
                system_info={},
                custom_metrics=custom_metrics or {}
            )
        
        start_metrics = self.current_metrics[step_name]
        end_time = time.time()
        
        # 현재 리소스 사용량 측정
        memory_after = self.get_current_memory_usage()
        cpu_current = self.get_current_cpu_usage()
        
        # 실행 시간 계산
        execution_time = end_time - start_metrics['start_time']
        
        # 메모리 사용량 계산
        memory_used = memory_after - start_metrics['memory_before']
        memory_peak = max(start_metrics['memory_before'], memory_after)
        
        # 성능 메트릭 객체 생성
        metrics = PerformanceMetrics(
            step_name=step_name,
            start_time=start_metrics['start_time'],
            end_time=end_time,
            execution_time=execution_time,
            memory_before=start_metrics['memory_before'],
            memory_after=memory_after,
            memory_peak=memory_peak,
            memory_used=memory_used,
            cpu_percent=cpu_current,
            system_info=start_metrics['system_info'],
            custom_metrics=custom_metrics or {}
        )
        
        # 히스토리에 추가
        self.metrics_history.append(metrics)
        
        # 현재 메트릭에서 제거
        del self.current_metrics[step_name]
        
        # 결과 로깅
        self.logger.info(f"📊 {step_name} 성능 모니터링 완료:")
        self.logger.info(f"  실행 시간: {execution_time:.3f}초")
        self.logger.info(f"  메모리 사용: {memory_used:+.2f} MB ({start_metrics['memory_before']:.2f} → {memory_after:.2f})")
        self.logger.info(f"  최대 메모리: {memory_peak:.2f} MB")
        self.logger.info(f"  CPU 사용률: {cpu_current:.1f}%")
        
        if custom_metrics:
            self.logger.info(f"  사용자 메트릭: {custom_metrics}")
        
        return metrics
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """
        연속적인 시스템 모니터링을 시작합니다.
        
        Parameters:
        -----------
        interval : float
            모니터링 간격 (초)
        """
        if not self.enable_monitoring or not PSUTIL_AVAILABLE:
            self.logger.warning("연속 모니터링을 사용할 수 없습니다.")
            return
        
        def monitor_loop():
            while not self.stop_monitoring:
                try:
                    timestamp = time.time()
                    memory_usage = self.get_current_memory_usage()
                    cpu_usage = self.get_current_cpu_usage()
                    
                    # 간단한 로깅 (상세한 로깅은 필요시 활성화)
                    if hasattr(self, '_last_log_time'):
                        if timestamp - self._last_log_time >= 10:  # 10초마다 로그
                            self.logger.debug(f"시스템 상태: 메모리 {memory_usage:.1f}MB, CPU {cpu_usage:.1f}%")
                            self._last_log_time = timestamp
                    else:
                        self._last_log_time = timestamp
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"연속 모니터링 오류: {e}")
                    break
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"연속 모니터링 시작 ({interval}초 간격)")
    
    def stop_continuous_monitoring(self):
        """연속적인 시스템 모니터링을 중지합니다."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring = True
            self.monitoring_thread.join(timeout=2.0)
            self.logger.info("연속 모니터링 중지")
    
    def benchmark_operation(self, operation: Callable, operation_name: str,
                          *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """
        특정 연산의 성능을 벤치마킹합니다.
        
        Parameters:
        -----------
        operation : Callable
            벤치마킹할 함수
        operation_name : str
            연산 이름
        *args, **kwargs
            함수 인자
            
        Returns:
        --------
        Tuple[Any, PerformanceMetrics]
            (함수 결과, 성능 메트릭)
        """
        self.start_step_monitoring(operation_name)
        
        try:
            result = operation(*args, **kwargs)
            
            # 연산별 사용자 정의 메트릭 추가
            custom_metrics = {}
            if hasattr(result, '__len__'):
                try:
                    custom_metrics['result_size'] = len(result)
                except:
                    pass
            
            metrics = self.end_step_monitoring(operation_name, custom_metrics)
            
            return result, metrics
            
        except Exception as e:
            # 오류 발생 시에도 메트릭 기록
            error_metrics = {'error': str(e)}
            metrics = self.end_step_monitoring(operation_name, error_metrics)
            raise
    
    def compare_performance(self, baseline_step: str, comparison_step: str) -> Dict[str, Any]:
        """
        두 단계의 성능을 비교합니다.
        
        Parameters:
        -----------
        baseline_step : str
            기준 단계 이름
        comparison_step : str
            비교 단계 이름
            
        Returns:
        --------
        Dict[str, Any]
            성능 비교 결과
        """
        baseline_metrics = None
        comparison_metrics = None
        
        # 해당 단계의 메트릭 찾기
        for metrics in self.metrics_history:
            if metrics.step_name == baseline_step and baseline_metrics is None:
                baseline_metrics = metrics
            elif metrics.step_name == comparison_step and comparison_metrics is None:
                comparison_metrics = metrics
        
        if not baseline_metrics or not comparison_metrics:
            return {'error': '비교할 메트릭을 찾을 수 없습니다'}
        
        # 성능 비교 계산
        time_ratio = comparison_metrics.execution_time / baseline_metrics.execution_time
        memory_ratio = comparison_metrics.memory_used / baseline_metrics.memory_used if baseline_metrics.memory_used != 0 else float('inf')
        
        comparison = {
            'baseline_step': baseline_step,
            'comparison_step': comparison_step,
            'execution_time': {
                'baseline': baseline_metrics.execution_time,
                'comparison': comparison_metrics.execution_time,
                'ratio': time_ratio,
                'faster': time_ratio < 1.0,
                'improvement': f"{(1 - time_ratio) * 100:+.1f}%" if time_ratio < 1.0 else f"{(time_ratio - 1) * 100:+.1f}% 느림"
            },
            'memory_usage': {
                'baseline': baseline_metrics.memory_used,
                'comparison': comparison_metrics.memory_used,
                'ratio': memory_ratio,
                'less_memory': memory_ratio < 1.0,
                'improvement': f"{(1 - memory_ratio) * 100:+.1f}%" if memory_ratio < 1.0 else f"{(memory_ratio - 1) * 100:+.1f}% 증가"
            }
        }
        
        return comparison
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        전체 성능 요약을 반환합니다.
        
        Returns:
        --------
        Dict[str, Any]
            성능 요약 정보
        """
        if not self.metrics_history:
            return {'message': '수집된 성능 메트릭이 없습니다'}
        
        total_time = sum(m.execution_time for m in self.metrics_history)
        total_memory = sum(m.memory_used for m in self.metrics_history)
        
        # 단계별 통계
        step_stats = {}
        for metrics in self.metrics_history:
            if metrics.step_name not in step_stats:
                step_stats[metrics.step_name] = {
                    'execution_times': [],
                    'memory_usage': [],
                    'cpu_usage': []
                }
            
            step_stats[metrics.step_name]['execution_times'].append(metrics.execution_time)
            step_stats[metrics.step_name]['memory_usage'].append(metrics.memory_used)
            step_stats[metrics.step_name]['cpu_usage'].append(metrics.cpu_percent)
        
        # 단계별 평균 계산
        step_averages = {}
        for step_name, stats in step_stats.items():
            step_averages[step_name] = {
                'avg_execution_time': sum(stats['execution_times']) / len(stats['execution_times']),
                'avg_memory_usage': sum(stats['memory_usage']) / len(stats['memory_usage']),
                'avg_cpu_usage': sum(stats['cpu_usage']) / len(stats['cpu_usage']),
                'run_count': len(stats['execution_times'])
            }
        
        # 가장 느린/빠른 단계 찾기
        slowest_step = max(step_averages.items(), key=lambda x: x[1]['avg_execution_time'])
        fastest_step = min(step_averages.items(), key=lambda x: x[1]['avg_execution_time'])
        
        # 가장 메모리 집약적인 단계 찾기
        memory_intensive_step = max(step_averages.items(), key=lambda x: x[1]['avg_memory_usage'])
        
        summary = {
            'overall_statistics': {
                'total_steps_measured': len(self.metrics_history),
                'unique_steps': len(step_averages),
                'total_execution_time': total_time,
                'total_memory_used': total_memory,
                'measurement_period': {
                    'start': min(m.start_time for m in self.metrics_history),
                    'end': max(m.end_time for m in self.metrics_history)
                }
            },
            'step_averages': step_averages,
            'performance_insights': {
                'slowest_step': {
                    'name': slowest_step[0],
                    'avg_time': slowest_step[1]['avg_execution_time']
                },
                'fastest_step': {
                    'name': fastest_step[0],
                    'avg_time': fastest_step[1]['avg_execution_time']
                },
                'memory_intensive_step': {
                    'name': memory_intensive_step[0],
                    'avg_memory': memory_intensive_step[1]['avg_memory_usage']
                }
            },
            'system_info': self.metrics_history[0].system_info if self.metrics_history else {}
        }
        
        return summary
    
    def save_performance_report(self, output_path: str = "performance_report.json"):
        """
        성능 보고서를 파일로 저장합니다.
        
        Parameters:
        -----------
        output_path : str
            출력 파일 경로
        """
        try:
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'detailed_metrics': [m.to_dict() for m in self.metrics_history],
                'monitoring_enabled': self.enable_monitoring,
                'psutil_available': PSUTIL_AVAILABLE
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 성능 보고서 저장: {output_file}")
            
        except Exception as e:
            self.logger.error(f"성능 보고서 저장 실패: {e}")
    
    def clear_history(self):
        """성능 메트릭 히스토리를 초기화합니다."""
        self.metrics_history.clear()
        self.current_metrics.clear()
        self.logger.info("성능 메트릭 히스토리 초기화")