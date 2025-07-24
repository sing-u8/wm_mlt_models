"""
요구사항 검증 스크립트

모든 시스템 요구사항, 의존성, 설정이 올바르게 충족되었는지 검증
"""

import os
import sys
import subprocess
import importlib
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import platform

# 프로젝트 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logger import LoggerMixin


@dataclass
class RequirementCheck:
    """개별 요구사항 검증 결과"""
    name: str
    required: bool  # 필수 여부
    installed: bool
    version_required: Optional[str] = None
    version_installed: Optional[str] = None
    status: str = "UNKNOWN"  # PASS, FAIL, WARNING
    details: str = ""


@dataclass
class RequirementsReport:
    """요구사항 검증 보고서"""
    system_info: Dict
    python_requirements: List[RequirementCheck]
    system_requirements: List[RequirementCheck]
    file_structure_checks: List[RequirementCheck]
    configuration_checks: List[RequirementCheck]
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    overall_status: str
    recommendations: List[str]


class RequirementsVerifier(LoggerMixin):
    """요구사항 검증기"""
    
    def __init__(self):
        self.logger = self.get_logger()
        
        # 필수 Python 패키지 정의
        self.required_packages = {
            'numpy': {'version': '>=1.19.0', 'required': True},
            'librosa': {'version': '>=0.8.0', 'required': True},
            'scikit-learn': {'version': '>=0.24.0', 'required': True},
            'pandas': {'version': '>=1.2.0', 'required': True},
            'matplotlib': {'version': '>=3.3.0', 'required': True},
            'seaborn': {'version': '>=0.11.0', 'required': True},
            'soundfile': {'version': '>=0.10.0', 'required': True},
            'psutil': {'version': '>=5.7.0', 'required': True},
            'tqdm': {'version': '>=4.50.0', 'required': True},
            'coremltools': {'version': '>=4.0', 'required': False},  # 선택적
            'GPUtil': {'version': None, 'required': False},  # 선택적
        }
        
        # 필수 디렉토리 구조
        self.required_directories = [
            'src',
            'src/audio',
            'src/data', 
            'src/ml',
            'src/utils',
            'src/config',
            'tests',
            'tests/unit',
            'tests/integration',
            'config',
            'logs',
            'results'
        ]
        
        # 필수 파일들
        self.required_files = [
            'requirements.txt',
            'main.py',
            'config.py',
            'src/__init__.py',
            'src/audio/feature_extraction.py',
            'src/data/pipeline.py',
            'src/ml/training.py',
            'src/ml/evaluation.py',
            'tests/conftest.py'
        ]
    
    def run_comprehensive_verification(self) -> RequirementsReport:
        """포괄적 요구사항 검증 실행"""
        self.logger.info("=== 요구사항 검증 시작 ===")
        
        # 시스템 정보 수집
        system_info = self._collect_system_info()
        
        # 각 카테고리별 검증
        python_reqs = self._verify_python_requirements()
        system_reqs = self._verify_system_requirements()
        file_checks = self._verify_file_structure()
        config_checks = self._verify_configuration()
        
        # 통계 계산
        all_checks = python_reqs + system_reqs + file_checks + config_checks
        total_checks = len(all_checks)
        passed_checks = sum(1 for c in all_checks if c.status == 'PASS')
        failed_checks = sum(1 for c in all_checks if c.status == 'FAIL')
        warning_checks = sum(1 for c in all_checks if c.status == 'WARNING')
        
        # 전체 상태 결정
        if failed_checks == 0:
            overall_status = 'PASS'
        elif failed_checks <= 2 and passed_checks > failed_checks:
            overall_status = 'PARTIAL_PASS'
        else:
            overall_status = 'FAIL'
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(all_checks)
        
        report = RequirementsReport(
            system_info=system_info,
            python_requirements=python_reqs,
            system_requirements=system_reqs,
            file_structure_checks=file_checks,
            configuration_checks=config_checks,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            overall_status=overall_status,
            recommendations=recommendations
        )
        
        self.logger.info(f"요구사항 검증 완료: {overall_status}")
        return report
    
    def _collect_system_info(self) -> Dict:
        """시스템 정보 수집"""
        try:
            import psutil
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'python_executable': sys.executable,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
                'current_directory': os.getcwd(),
                'python_path': sys.path[:3]  # 처음 3개만
            }
        except Exception as e:
            self.logger.warning(f"시스템 정보 수집 실패: {e}")
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'error': str(e)
            }
    
    def _verify_python_requirements(self) -> List[RequirementCheck]:
        """Python 패키지 요구사항 검증"""
        self.logger.info("Python 패키지 요구사항 검증 중...")
        
        checks = []
        
        for package_name, info in self.required_packages.items():
            check = RequirementCheck(
                name=f"python-{package_name}",
                required=info['required'],
                installed=False,
                version_required=info.get('version')
            )
            
            try:
                # 패키지 임포트 시도
                module = importlib.import_module(package_name)
                check.installed = True
                
                # 버전 확인
                version = getattr(module, '__version__', None)
                if version:
                    check.version_installed = version
                    
                    # 버전 요구사항 확인
                    if check.version_required:
                        if self._check_version_requirement(version, check.version_required):
                            check.status = 'PASS'
                            check.details = f"버전 {version} 설치됨"
                        else:
                            check.status = 'FAIL' if check.required else 'WARNING'
                            check.details = f"버전 불일치: {version} (요구: {check.version_required})"
                    else:
                        check.status = 'PASS'
                        check.details = f"버전 {version} 설치됨"
                else:
                    check.status = 'PASS' if check.required else 'WARNING'
                    check.details = "설치됨 (버전 확인 불가)"
                    
            except ImportError as e:
                check.installed = False
                check.status = 'FAIL' if check.required else 'WARNING'
                check.details = f"설치되지 않음: {str(e)}"
            
            checks.append(check)
        
        return checks
    
    def _check_version_requirement(self, installed_version: str, requirement: str) -> bool:
        """버전 요구사항 확인"""
        try:
            from packaging import version
            
            # 요구사항 파싱 (>=1.0.0 형태)
            if requirement.startswith('>='):
                required_version = requirement[2:]
                return version.parse(installed_version) >= version.parse(required_version)
            elif requirement.startswith('=='):
                required_version = requirement[2:]
                return version.parse(installed_version) == version.parse(required_version)
            elif requirement.startswith('>'):
                required_version = requirement[1:]
                return version.parse(installed_version) > version.parse(required_version)
            else:
                # 단순 버전 비교
                return version.parse(installed_version) >= version.parse(requirement)
                
        except Exception:
            # packaging 모듈이 없거나 파싱 실패시 단순 문자열 비교
            return installed_version >= requirement
    
    def _verify_system_requirements(self) -> List[RequirementCheck]:
        """시스템 요구사항 검증"""
        self.logger.info("시스템 요구사항 검증 중...")
        
        checks = []
        
        try:
            import psutil
            
            # CPU 요구사항
            cpu_count = psutil.cpu_count()
            cpu_check = RequirementCheck(
                name="system-cpu",
                required=True,
                installed=True,
                version_installed=f"{cpu_count} cores"
            )
            
            if cpu_count >= 2:
                cpu_check.status = 'PASS'
                cpu_check.details = f"{cpu_count}개 코어 (최소 2개 필요)"
            else:
                cpu_check.status = 'WARNING'
                cpu_check.details = f"{cpu_count}개 코어 (최소 2개 권장)"
            
            checks.append(cpu_check)
            
            # 메모리 요구사항
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_check = RequirementCheck(
                name="system-memory",
                required=True,
                installed=True,
                version_installed=f"{memory_gb:.1f} GB"
            )
            
            if memory_gb >= 4.0:
                memory_check.status = 'PASS'
                memory_check.details = f"{memory_gb:.1f}GB (최소 2GB 필요)"
            elif memory_gb >= 2.0:
                memory_check.status = 'WARNING'
                memory_check.details = f"{memory_gb:.1f}GB (4GB 권장)"
            else:
                memory_check.status = 'FAIL'
                memory_check.details = f"{memory_gb:.1f}GB (최소 2GB 필요)"
            
            checks.append(memory_check)
            
            # 디스크 공간 요구사항
            disk_free_gb = psutil.disk_usage('.').free / (1024**3)
            disk_check = RequirementCheck(
                name="system-disk",
                required=True,
                installed=True,
                version_installed=f"{disk_free_gb:.1f} GB free"
            )
            
            if disk_free_gb >= 5.0:
                disk_check.status = 'PASS'
                disk_check.details = f"{disk_free_gb:.1f}GB 사용 가능 (최소 1GB 필요)"
            elif disk_free_gb >= 1.0:
                disk_check.status = 'WARNING'
                disk_check.details = f"{disk_free_gb:.1f}GB 사용 가능 (5GB 권장)"
            else:
                disk_check.status = 'FAIL'
                disk_check.details = f"{disk_free_gb:.1f}GB 사용 가능 (최소 1GB 필요)"
            
            checks.append(disk_check)
            
        except Exception as e:
            error_check = RequirementCheck(
                name="system-info",
                required=True,
                installed=False,
                status='FAIL',
                details=f"시스템 정보 수집 실패: {str(e)}"
            )
            checks.append(error_check)
        
        # Python 버전 확인
        python_version = platform.python_version()
        python_check = RequirementCheck(
            name="python-version",
            required=True,
            installed=True,
            version_installed=python_version,
            version_required=">=3.7.0"
        )
        
        if self._check_version_requirement(python_version, ">=3.7.0"):
            python_check.status = 'PASS'
            python_check.details = f"Python {python_version}"
        else:
            python_check.status = 'FAIL'
            python_check.details = f"Python {python_version} (최소 3.7.0 필요)"
        
        checks.append(python_check)
        
        return checks
    
    def _verify_file_structure(self) -> List[RequirementCheck]:
        """파일 구조 요구사항 검증"""
        self.logger.info("파일 구조 요구사항 검증 중...")
        
        checks = []
        
        # 디렉토리 검증
        for directory in self.required_directories:
            check = RequirementCheck(
                name=f"dir-{directory}",
                required=True,
                installed=os.path.exists(directory)
            )
            
            if check.installed:
                check.status = 'PASS'
                check.details = f"디렉토리 존재: {directory}"
            else:
                check.status = 'FAIL'
                check.details = f"디렉토리 없음: {directory}"
            
            checks.append(check)
        
        # 파일 검증
        for file_path in self.required_files:
            check = RequirementCheck(
                name=f"file-{os.path.basename(file_path)}",
                required=True,
                installed=os.path.exists(file_path)
            )
            
            if check.installed:
                # 파일 크기 확인
                try:
                    file_size = os.path.getsize(file_path)
                    check.status = 'PASS'
                    check.details = f"파일 존재: {file_path} ({file_size} bytes)"
                except Exception as e:
                    check.status = 'WARNING'
                    check.details = f"파일 존재하지만 접근 불가: {file_path}"
            else:
                check.status = 'FAIL'
                check.details = f"파일 없음: {file_path}"
            
            checks.append(check)
        
        # 데이터 디렉토리 확인 (선택적)
        data_dirs = ['data', 'data/raw', 'data/noise']
        for data_dir in data_dirs:
            check = RequirementCheck(
                name=f"data-{data_dir}",
                required=False,
                installed=os.path.exists(data_dir)
            )
            
            if check.installed:
                check.status = 'PASS'
                check.details = f"데이터 디렉토리 존재: {data_dir}"
            else:
                check.status = 'WARNING'
                check.details = f"데이터 디렉토리 없음: {data_dir} (런타임에 생성 가능)"
            
            checks.append(check)
        
        return checks
    
    def _verify_configuration(self) -> List[RequirementCheck]:
        """설정 파일 요구사항 검증"""
        self.logger.info("설정 요구사항 검증 중...")
        
        checks = []
        
        # config.py 검증
        config_check = RequirementCheck(
            name="config-file",
            required=True,
            installed=False
        )
        
        try:
            # config.py 임포트 시도
            import config
            config_check.installed = True
            
            # DEFAULT_CONFIG 존재 확인
            if hasattr(config, 'DEFAULT_CONFIG'):
                config_check.status = 'PASS'
                config_check.details = "config.py와 DEFAULT_CONFIG 정상"
            else:
                config_check.status = 'FAIL'
                config_check.details = "DEFAULT_CONFIG 없음"
                
        except ImportError as e:
            config_check.status = 'FAIL'
            config_check.details = f"config.py 임포트 실패: {str(e)}"
        
        checks.append(config_check)
        
        # requirements.txt 검증
        req_file_check = RequirementCheck(
            name="requirements-txt",
            required=True,
            installed=os.path.exists('requirements.txt')
        )
        
        if req_file_check.installed:
            try:
                with open('requirements.txt', 'r') as f:
                    requirements = f.read()
                    req_count = len([line for line in requirements.split('\n') 
                                   if line.strip() and not line.startswith('#')])
                
                req_file_check.status = 'PASS'
                req_file_check.details = f"requirements.txt 존재 ({req_count}개 패키지)"
            except Exception as e:
                req_file_check.status = 'WARNING'
                req_file_check.details = f"requirements.txt 읽기 실패: {str(e)}"
        else:
            req_file_check.status = 'FAIL'
            req_file_check.details = "requirements.txt 없음"
        
        checks.append(req_file_check)
        
        # 로그 디렉토리 권한 확인
        log_check = RequirementCheck(
            name="logs-writable",
            required=True,
            installed=False
        )
        
        try:
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 쓰기 권한 테스트
            test_file = os.path.join(log_dir, 'write_test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            log_check.installed = True
            log_check.status = 'PASS'
            log_check.details = "로그 디렉토리 쓰기 가능"
            
        except Exception as e:
            log_check.status = 'FAIL'
            log_check.details = f"로그 디렉토리 쓰기 불가: {str(e)}"
        
        checks.append(log_check)
        
        return checks
    
    def _generate_recommendations(self, all_checks: List[RequirementCheck]) -> List[str]:
        """검증 결과 기반 권장사항 생성"""
        recommendations = []
        
        failed_checks = [c for c in all_checks if c.status == 'FAIL']
        warning_checks = [c for c in all_checks if c.status == 'WARNING']
        
        # 실패한 검사 기반 권장사항
        python_failures = [c for c in failed_checks if c.name.startswith('python-')]
        if python_failures:
            missing_packages = [c.name.replace('python-', '') for c in python_failures]
            recommendations.append(f"필수 Python 패키지 설치: pip install {' '.join(missing_packages)}")
        
        system_failures = [c for c in failed_checks if c.name.startswith('system-')]
        if system_failures:
            recommendations.append("시스템 리소스(CPU, 메모리, 디스크)를 확인하고 업그레이드하세요")
        
        file_failures = [c for c in failed_checks if c.name.startswith(('dir-', 'file-'))]
        if file_failures:
            recommendations.append("누락된 디렉토리나 파일을 생성하거나 복원하세요")
        
        config_failures = [c for c in failed_checks if c.name.startswith('config-')]
        if config_failures:
            recommendations.append("설정 파일(config.py, requirements.txt)을 확인하세요")
        
        # 경고 기반 권장사항
        if warning_checks:
            recommendations.append("경고 항목들을 검토하여 최적 성능을 위해 개선하세요")
        
        # 일반적인 권장사항
        if not failed_checks:
            recommendations.append("모든 요구사항이 충족되었습니다!")
        else:
            recommendations.append("실패한 항목들을 먼저 해결한 후 시스템을 다시 테스트하세요")
        
        return recommendations
    
    def save_report(self, report: RequirementsReport, output_file: str = "requirements_report.json"):
        """요구사항 보고서 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"요구사항 보고서 저장: {output_file}")
            
            # 텍스트 요약도 저장
            summary_file = output_file.replace('.json', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("수박 소리 분류 시스템 요구사항 검증 보고서\n")
                f.write("="*50 + "\n\n")
                f.write(f"전체 상태: {report.overall_status}\n")
                f.write(f"검증 항목: {report.total_checks}개\n")
                f.write(f"통과: {report.passed_checks}개\n")
                f.write(f"실패: {report.failed_checks}개\n")
                f.write(f"경고: {report.warning_checks}개\n\n")
                
                f.write("상세 결과:\n")
                f.write("-" * 30 + "\n")
                
                # 카테고리별 결과
                categories = [
                    ("Python 패키지", report.python_requirements),
                    ("시스템 요구사항", report.system_requirements),
                    ("파일 구조", report.file_structure_checks),
                    ("설정", report.configuration_checks)
                ]
                
                for category_name, checks in categories:
                    f.write(f"\n{category_name}:\n")
                    for check in checks:
                        status_symbol = "✅" if check.status == 'PASS' else "❌" if check.status == 'FAIL' else "⚠️"
                        f.write(f"  {status_symbol} {check.name}: {check.details}\n")
                
                if report.recommendations:
                    f.write("\n권장사항:\n")
                    f.write("-" * 30 + "\n")
                    for rec in report.recommendations:
                        f.write(f"• {rec}\n")
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='수박 소리 분류 시스템 요구사항 검증')
    parser.add_argument('--output', default='requirements_report.json', help='출력 파일명')
    
    args = parser.parse_args()
    
    # 검증기 생성 및 실행
    verifier = RequirementsVerifier()
    report = verifier.run_comprehensive_verification()
    
    # 보고서 저장
    verifier.save_report(report, args.output)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"수박 소리 분류 시스템 요구사항 검증 결과")
    print(f"{'='*60}")
    print(f"전체 상태: {report.overall_status}")
    print(f"검증 항목: 통과 {report.passed_checks}, 실패 {report.failed_checks}, 경고 {report.warning_checks}")
    
    # 실패한 항목 표시
    failed_checks = [c for c in (report.python_requirements + report.system_requirements + 
                                report.file_structure_checks + report.configuration_checks) 
                    if c.status == 'FAIL']
    
    if failed_checks:
        print(f"\n❌ 실패한 항목들:")
        for check in failed_checks:
            print(f"   • {check.name}: {check.details}")
    
    # 권장사항 표시
    if report.recommendations:
        print(f"\n💡 권장사항:")
        for rec in report.recommendations:
            print(f"   • {rec}")
    
    print(f"\n상세 보고서: {args.output}")
    
    # 종료 코드
    return 0 if report.overall_status in ['PASS', 'PARTIAL_PASS'] else 1


if __name__ == "__main__":
    sys.exit(main())