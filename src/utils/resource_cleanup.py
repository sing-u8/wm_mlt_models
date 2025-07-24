"""
리소스 정리 및 최적화 유틸리티

임시 파일, 캐시, 메모리 사용량을 최적화하고 시스템 리소스를 정리하는 유틸리티
"""

import os
import sys
import gc
import shutil
import tempfile
import time
import psutil
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .logger import LoggerMixin


@dataclass
class CleanupResult:
    """정리 결과"""
    category: str
    files_removed: int
    space_freed_mb: float
    execution_time: float
    success: bool
    details: str


@dataclass
class ResourceReport:
    """리소스 사용 보고서"""
    timestamp: str
    memory_before_mb: float
    memory_after_mb: float
    memory_freed_mb: float
    disk_space_freed_mb: float
    temp_files_removed: int
    cache_files_removed: int
    log_files_cleaned: int
    total_execution_time: float
    cleanup_results: List[CleanupResult]


class ResourceCleanupManager(LoggerMixin):
    """리소스 정리 관리자"""
    
    def __init__(self):
        self.logger = self.get_logger()
        
        # 정리 대상 패턴들
        self.cleanup_patterns = {
            'temp_files': [
                '**/*.tmp',
                '**/*.temp',
                '**/temp_*',
                '**/__pycache__/**',
                '**/*.pyc',
                '**/*~',
                '**/.*~'
            ],
            'cache_files': [
                '**/.cache/**',
                '**/cache/**',
                '**/*.cache',
                '**/librosa_cache/**',
                '**/.pytest_cache/**'
            ],
            'log_files': [
                'logs/*.log.*',  # 로테이션된 로그만
                'logs/old_*.log',
                '**/*.log.old'
            ],
            'output_files': [
                'results/temp_*',
                'output/temp_*',
                '**/validation_*',
                '**/benchmark_*'
            ]
        }
        
        # 보호할 패턴들 (삭제하지 않음)
        self.protected_patterns = [
            'logs/current.log',
            'logs/main.log',
            'config/**',
            'src/**',
            '.git/**',
            '**/.gitignore',
            '**/requirements.txt',
            '**/README.md'
        ]
    
    def run_comprehensive_cleanup(self, 
                                preserve_recent_hours: int = 24,
                                dry_run: bool = False) -> ResourceReport:
        """
        포괄적 리소스 정리 실행
        
        Args:
            preserve_recent_hours: 최근 N시간 내 파일 보호
            dry_run: 실제 삭제 없이 시뮬레이션만
            
        Returns:
            정리 결과 보고서
        """
        self.logger.info(f"포괄적 리소스 정리 시작 (dry_run={dry_run})")
        start_time = time.time()
        
        # 시작 메모리 사용량
        memory_before = self._get_memory_usage_mb()
        
        cleanup_results = []
        
        # 1. 임시 파일 정리
        temp_result = self._cleanup_temp_files(preserve_recent_hours, dry_run)
        cleanup_results.append(temp_result)
        
        # 2. 캐시 파일 정리
        cache_result = self._cleanup_cache_files(preserve_recent_hours, dry_run)
        cleanup_results.append(cache_result)
        
        # 3. 로그 파일 정리 (오래된 것만)
        log_result = self._cleanup_old_logs(preserve_recent_hours, dry_run)
        cleanup_results.append(log_result)
        
        # 4. 출력 파일 정리
        output_result = self._cleanup_output_files(preserve_recent_hours, dry_run)
        cleanup_results.append(output_result)
        
        # 5. 시스템 임시 디렉토리 정리
        system_temp_result = self._cleanup_system_temp(dry_run)
        cleanup_results.append(system_temp_result)
        
        # 6. 메모리 최적화
        memory_result = self._optimize_memory()
        cleanup_results.append(memory_result)
        
        # 종료 메모리 사용량
        memory_after = self._get_memory_usage_mb()
        
        total_time = time.time() - start_time
        
        # 보고서 생성
        report = ResourceReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_freed_mb=memory_before - memory_after,
            disk_space_freed_mb=sum(r.space_freed_mb for r in cleanup_results),
            temp_files_removed=sum(r.files_removed for r in cleanup_results 
                                 if r.category in ['temp_files', 'system_temp']),
            cache_files_removed=cleanup_results[1].files_removed if len(cleanup_results) > 1 else 0,
            log_files_cleaned=cleanup_results[2].files_removed if len(cleanup_results) > 2 else 0,
            total_execution_time=total_time,
            cleanup_results=cleanup_results
        )
        
        self.logger.info(f"리소스 정리 완료: {report.disk_space_freed_mb:.1f}MB 확보, "
                        f"{report.memory_freed_mb:.1f}MB 메모리 해제")
        
        return report
    
    def _cleanup_temp_files(self, preserve_hours: int, dry_run: bool) -> CleanupResult:
        """임시 파일 정리"""
        self.logger.info("임시 파일 정리 중...")
        start_time = time.time()
        
        files_removed = 0
        space_freed = 0.0
        errors = []
        
        try:
            cutoff_time = time.time() - (preserve_hours * 3600)
            
            for pattern in self.cleanup_patterns['temp_files']:
                try:
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_path in files:
                        try:
                            # 보호된 패턴 확인
                            if self._is_protected_file(file_path):
                                continue
                            
                            # 파일 존재 및 수정 시간 확인
                            if os.path.exists(file_path):
                                stat = os.stat(file_path)
                                
                                # 최근 파일 보호
                                if stat.st_mtime > cutoff_time:
                                    continue
                                
                                file_size = stat.st_size
                                
                                if not dry_run:
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                    elif os.path.isdir(file_path):
                                        shutil.rmtree(file_path)
                                
                                files_removed += 1
                                space_freed += file_size / (1024 * 1024)  # MB
                        
                        except Exception as e:
                            errors.append(f"파일 삭제 실패 {file_path}: {str(e)}")
                
                except Exception as e:
                    errors.append(f"패턴 처리 실패 {pattern}: {str(e)}")
        
        except Exception as e:
            errors.append(f"임시 파일 정리 실패: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return CleanupResult(
            category='temp_files',
            files_removed=files_removed,
            space_freed_mb=space_freed,
            execution_time=execution_time,
            success=len(errors) == 0,
            details=f"임시 파일 {files_removed}개 정리" + (f", 오류 {len(errors)}개" if errors else "")
        )
    
    def _cleanup_cache_files(self, preserve_hours: int, dry_run: bool) -> CleanupResult:
        """캐시 파일 정리"""
        self.logger.info("캐시 파일 정리 중...")
        start_time = time.time()
        
        files_removed = 0
        space_freed = 0.0
        errors = []
        
        try:
            cutoff_time = time.time() - (preserve_hours * 3600)
            
            for pattern in self.cleanup_patterns['cache_files']:
                try:
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_path in files:
                        try:
                            if self._is_protected_file(file_path):
                                continue
                            
                            if os.path.exists(file_path):
                                stat = os.stat(file_path)
                                
                                # 최근 캐시는 보존 (더 긴 보존 기간)
                                if stat.st_mtime > cutoff_time - 3600:  # 추가 1시간 보존
                                    continue
                                
                                if os.path.isfile(file_path):
                                    file_size = stat.st_size
                                    if not dry_run:
                                        os.remove(file_path)
                                    files_removed += 1
                                    space_freed += file_size / (1024 * 1024)
                                
                                elif os.path.isdir(file_path):
                                    dir_size = self._get_directory_size(file_path)
                                    if not dry_run:
                                        shutil.rmtree(file_path)
                                    files_removed += 1
                                    space_freed += dir_size / (1024 * 1024)
                        
                        except Exception as e:
                            errors.append(f"캐시 삭제 실패 {file_path}: {str(e)}")
                
                except Exception as e:
                    errors.append(f"캐시 패턴 처리 실패 {pattern}: {str(e)}")
        
        except Exception as e:
            errors.append(f"캐시 정리 실패: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return CleanupResult(
            category='cache_files',
            files_removed=files_removed,
            space_freed_mb=space_freed,
            execution_time=execution_time,
            success=len(errors) == 0,
            details=f"캐시 파일 {files_removed}개 정리" + (f", 오류 {len(errors)}개" if errors else "")
        )
    
    def _cleanup_old_logs(self, preserve_hours: int, dry_run: bool) -> CleanupResult:
        """오래된 로그 파일 정리"""
        self.logger.info("오래된 로그 파일 정리 중...")
        start_time = time.time()
        
        files_removed = 0
        space_freed = 0.0
        errors = []
        
        try:
            # 로그는 더 오래 보존 (7일)
            cutoff_time = time.time() - (7 * 24 * 3600)
            
            for pattern in self.cleanup_patterns['log_files']:
                try:
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_path in files:
                        try:
                            if self._is_protected_file(file_path):
                                continue
                            
                            if os.path.exists(file_path) and os.path.isfile(file_path):
                                stat = os.stat(file_path)
                                
                                if stat.st_mtime < cutoff_time:
                                    file_size = stat.st_size
                                    
                                    if not dry_run:
                                        os.remove(file_path)
                                    
                                    files_removed += 1
                                    space_freed += file_size / (1024 * 1024)
                        
                        except Exception as e:
                            errors.append(f"로그 삭제 실패 {file_path}: {str(e)}")
                
                except Exception as e:
                    errors.append(f"로그 패턴 처리 실패 {pattern}: {str(e)}")
        
        except Exception as e:
            errors.append(f"로그 정리 실패: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return CleanupResult(
            category='log_files',
            files_removed=files_removed,
            space_freed_mb=space_freed,
            execution_time=execution_time,
            success=len(errors) == 0,
            details=f"로그 파일 {files_removed}개 정리" + (f", 오류 {len(errors)}개" if errors else "")
        )
    
    def _cleanup_output_files(self, preserve_hours: int, dry_run: bool) -> CleanupResult:
        """출력 파일 정리"""
        self.logger.info("임시 출력 파일 정리 중...")
        start_time = time.time()
        
        files_removed = 0
        space_freed = 0.0
        errors = []
        
        try:
            cutoff_time = time.time() - (preserve_hours * 3600)
            
            for pattern in self.cleanup_patterns['output_files']:
                try:
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_path in files:
                        try:
                            if self._is_protected_file(file_path):
                                continue
                            
                            if os.path.exists(file_path):
                                stat = os.stat(file_path)
                                
                                if stat.st_mtime < cutoff_time:
                                    if os.path.isfile(file_path):
                                        file_size = stat.st_size
                                        if not dry_run:
                                            os.remove(file_path)
                                        files_removed += 1
                                        space_freed += file_size / (1024 * 1024)
                                    
                                    elif os.path.isdir(file_path):
                                        dir_size = self._get_directory_size(file_path)
                                        if not dry_run:
                                            shutil.rmtree(file_path)
                                        files_removed += 1
                                        space_freed += dir_size / (1024 * 1024)
                        
                        except Exception as e:
                            errors.append(f"출력 파일 삭제 실패 {file_path}: {str(e)}")
                
                except Exception as e:
                    errors.append(f"출력 패턴 처리 실패 {pattern}: {str(e)}")
        
        except Exception as e:
            errors.append(f"출력 파일 정리 실패: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return CleanupResult(
            category='output_files',
            files_removed=files_removed,
            space_freed_mb=space_freed,
            execution_time=execution_time,
            success=len(errors) == 0,
            details=f"출력 파일 {files_removed}개 정리" + (f", 오류 {len(errors)}개" if errors else "")
        )
    
    def _cleanup_system_temp(self, dry_run: bool) -> CleanupResult:
        """시스템 임시 디렉토리 정리"""
        self.logger.info("시스템 임시 디렉토리 정리 중...")
        start_time = time.time()
        
        files_removed = 0
        space_freed = 0.0
        errors = []
        
        try:
            temp_dirs = [tempfile.gettempdir()]
            
            # 추가 임시 디렉토리 (플랫폼별)
            if os.name == 'posix':
                temp_dirs.extend(['/tmp', '/var/tmp'])
            elif os.name == 'nt':
                temp_dirs.extend([os.environ.get('TEMP', ''), os.environ.get('TMP', '')])
            
            # 프로젝트 관련 임시 파일만 정리
            project_prefixes = ['wm_', 'watermelon_', 'audio_', 'ml_', 'temp_']
            
            for temp_dir in temp_dirs:
                if not temp_dir or not os.path.exists(temp_dir):
                    continue
                
                try:
                    for item in os.listdir(temp_dir):
                        if any(item.startswith(prefix) for prefix in project_prefixes):
                            item_path = os.path.join(temp_dir, item)
                            
                            try:
                                if os.path.isfile(item_path):
                                    file_size = os.path.getsize(item_path)
                                    if not dry_run:
                                        os.remove(item_path)
                                    files_removed += 1
                                    space_freed += file_size / (1024 * 1024)
                                
                                elif os.path.isdir(item_path):
                                    dir_size = self._get_directory_size(item_path)
                                    if not dry_run:
                                        shutil.rmtree(item_path)
                                    files_removed += 1
                                    space_freed += dir_size / (1024 * 1024)
                            
                            except Exception as e:
                                errors.append(f"시스템 임시 파일 삭제 실패 {item_path}: {str(e)}")
                
                except Exception as e:
                    errors.append(f"임시 디렉토리 접근 실패 {temp_dir}: {str(e)}")
        
        except Exception as e:
            errors.append(f"시스템 임시 정리 실패: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return CleanupResult(
            category='system_temp',
            files_removed=files_removed,
            space_freed_mb=space_freed,
            execution_time=execution_time,
            success=len(errors) == 0,
            details=f"시스템 임시 파일 {files_removed}개 정리" + (f", 오류 {len(errors)}개" if errors else "")
        )
    
    def _optimize_memory(self) -> CleanupResult:
        """메모리 최적화"""
        self.logger.info("메모리 최적화 중...")
        start_time = time.time()
        
        try:
            memory_before = self._get_memory_usage_mb()
            
            # 가비지 컬렉션 강제 실행
            collected = 0
            for generation in range(3):
                collected += gc.collect()
            
            # 캐시 정리 (가능한 경우)
            try:
                # numpy 캐시 정리
                import numpy as np
                if hasattr(np, 'get_printoptions'):
                    pass  # numpy 내부 캐시는 직접 정리하기 어려움
            except ImportError:
                pass
            
            try:
                # librosa 캐시 정리
                import librosa
                if hasattr(librosa, 'cache'):
                    librosa.cache.cache.clear()
            except ImportError:
                pass
            
            memory_after = self._get_memory_usage_mb()
            memory_freed = memory_before - memory_after
            
            execution_time = time.time() - start_time
            
            return CleanupResult(
                category='memory_optimization',
                files_removed=collected,
                space_freed_mb=memory_freed,
                execution_time=execution_time,
                success=True,
                details=f"메모리 {memory_freed:.1f}MB 해제, GC 객체 {collected}개 정리"
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return CleanupResult(
                category='memory_optimization',
                files_removed=0,
                space_freed_mb=0.0,
                execution_time=execution_time,
                success=False,
                details=f"메모리 최적화 실패: {str(e)}"
            )
    
    def _is_protected_file(self, file_path: str) -> bool:
        """보호된 파일 확인"""
        for pattern in self.protected_patterns:
            if glob.fnmatch.fnmatch(file_path, pattern):
                return True
        return False
    
    def _get_directory_size(self, directory: str) -> int:
        """디렉토리 크기 계산 (bytes)"""
        total_size = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        continue
        except Exception:
            pass
        return total_size
    
    def _get_memory_usage_mb(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def generate_cleanup_report(self, report: ResourceReport, output_file: str = None):
        """정리 보고서 생성"""
        if output_file is None:
            output_file = f"cleanup_report_{int(time.time())}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("수박 소리 분류 시스템 리소스 정리 보고서\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"정리 시간: {report.timestamp}\n")
                f.write(f"총 실행 시간: {report.total_execution_time:.2f}초\n\n")
                
                f.write("리소스 사용량 변화:\n")
                f.write(f"  메모리 사용량: {report.memory_before_mb:.1f}MB → {report.memory_after_mb:.1f}MB\n")
                f.write(f"  메모리 해제: {report.memory_freed_mb:.1f}MB\n")
                f.write(f"  디스크 공간 확보: {report.disk_space_freed_mb:.1f}MB\n\n")
                
                f.write("정리 결과:\n")
                f.write(f"  임시 파일: {report.temp_files_removed}개 제거\n")
                f.write(f"  캐시 파일: {report.cache_files_removed}개 제거\n")
                f.write(f"  로그 파일: {report.log_files_cleaned}개 정리\n\n")
                
                f.write("카테고리별 상세 결과:\n")
                f.write("-" * 30 + "\n")
                for result in report.cleanup_results:
                    status = "✅" if result.success else "❌"
                    f.write(f"{status} {result.category}:\n")
                    f.write(f"  파일/객체: {result.files_removed}개\n")
                    f.write(f"  공간 확보: {result.space_freed_mb:.1f}MB\n")
                    f.write(f"  실행 시간: {result.execution_time:.2f}초\n")
                    f.write(f"  세부사항: {result.details}\n\n")
            
            self.logger.info(f"정리 보고서 저장: {output_file}")
            
        except Exception as e:
            self.logger.error(f"보고서 생성 실패: {e}")
    
    def get_disk_usage_summary(self) -> Dict:
        """디스크 사용량 요약"""
        try:
            current_dir = os.getcwd()
            
            # 주요 디렉토리별 사용량
            directories = {
                'src': 'src',
                'data': 'data',
                'results': 'results',
                'logs': 'logs',
                'tests': 'tests',
                '__pycache__': '**/__pycache__'
            }
            
            usage = {}
            total_size = 0
            
            for name, path in directories.items():
                if os.path.exists(path):
                    if '**' in path:
                        # glob 패턴
                        size = 0
                        for match in glob.glob(path, recursive=True):
                            if os.path.isdir(match):
                                size += self._get_directory_size(match)
                    else:
                        # 단일 디렉토리
                        size = self._get_directory_size(path)
                    
                    usage[name] = size / (1024 * 1024)  # MB
                    total_size += size
                else:
                    usage[name] = 0.0
            
            usage['total'] = total_size / (1024 * 1024)
            return usage
            
        except Exception as e:
            self.logger.error(f"디스크 사용량 계산 실패: {e}")
            return {}


# 편의 함수들
def quick_cleanup(preserve_hours: int = 24, dry_run: bool = False) -> ResourceReport:
    """빠른 리소스 정리"""
    manager = ResourceCleanupManager()
    return manager.run_comprehensive_cleanup(preserve_hours, dry_run)

def cleanup_temp_files_only(dry_run: bool = False) -> CleanupResult:
    """임시 파일만 정리"""
    manager = ResourceCleanupManager()
    return manager._cleanup_temp_files(24, dry_run)

def optimize_memory() -> CleanupResult:
    """메모리만 최적화"""
    manager = ResourceCleanupManager()
    return manager._optimize_memory()

def get_system_usage() -> Dict:
    """시스템 사용량 확인"""
    manager = ResourceCleanupManager()
    return manager.get_disk_usage_summary()


# CLI 인터페이스
def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='수박 소리 분류 시스템 리소스 정리')
    parser.add_argument('--preserve-hours', type=int, default=24, 
                       help='최근 N시간 내 파일 보호 (기본: 24)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='실제 삭제 없이 시뮬레이션만')
    parser.add_argument('--report-file', help='보고서 파일명')
    parser.add_argument('--temp-only', action='store_true', 
                       help='임시 파일만 정리')
    parser.add_argument('--memory-only', action='store_true', 
                       help='메모리만 최적화')
    parser.add_argument('--usage', action='store_true', 
                       help='디스크 사용량만 확인')
    
    args = parser.parse_args()
    
    if args.usage:
        # 사용량만 확인
        usage = get_system_usage()
        print("디스크 사용량 요약:")
        for name, size_mb in usage.items():
            print(f"  {name}: {size_mb:.1f} MB")
        return
    
    if args.memory_only:
        # 메모리만 최적화
        result = optimize_memory()
        print(f"메모리 최적화: {result.details}")
        return
    
    if args.temp_only:
        # 임시 파일만 정리
        result = cleanup_temp_files_only(args.dry_run)
        print(f"임시 파일 정리: {result.details}")
        return
    
    # 포괄적 정리
    manager = ResourceCleanupManager()
    report = manager.run_comprehensive_cleanup(args.preserve_hours, args.dry_run)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"리소스 정리 결과")
    print(f"{'='*50}")
    print(f"실행 시간: {report.total_execution_time:.2f}초")
    print(f"메모리 해제: {report.memory_freed_mb:.1f}MB")
    print(f"디스크 공간 확보: {report.disk_space_freed_mb:.1f}MB")
    print(f"임시 파일: {report.temp_files_removed}개 제거")
    print(f"캐시 파일: {report.cache_files_removed}개 제거")
    
    # 보고서 생성
    if args.report_file:
        manager.generate_cleanup_report(report, args.report_file)
        print(f"상세 보고서: {args.report_file}")


if __name__ == "__main__":
    main()