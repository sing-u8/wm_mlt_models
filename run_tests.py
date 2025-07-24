#!/usr/bin/env python3
"""
테스트 실행 스크립트

수박 소리 분류 시스템의 전체 테스트 스위트를 실행하고 결과를 보고합니다.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_command(cmd, description=""):
    """명령어 실행 및 결과 반환"""
    print(f"\n{'='*60}")
    print(f"실행: {description}")
    print(f"명령어: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(__file__)
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except Exception as e:
        print(f"명령어 실행 실패: {e}")
        return False, "", str(e)


def run_unit_tests():
    """단위 테스트 실행"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-m", "unit",
        "--tb=short",
        "-v"
    ]
    
    return run_command(cmd, "단위 테스트 실행")


def run_integration_tests():
    """통합 테스트 실행"""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/",
        "-m", "integration",
        "--tb=short",
        "-v"
    ]
    
    return run_command(cmd, "통합 테스트 실행")


def run_performance_tests():
    """성능 테스트 실행"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-m", "performance",
        "--tb=short",
        "-v"
    ]
    
    return run_command(cmd, "성능 테스트 실행")


def run_all_tests(exclude_slow=False):
    """모든 테스트 실행"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    if exclude_slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "전체 테스트 실행")


def run_coverage_report():
    """코드 커버리지 보고서 생성"""
    # 커버리지와 함께 테스트 실행
    cmd_test = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "-m", "not slow"  # 느린 테스트 제외
    ]
    
    success, stdout, stderr = run_command(cmd_test, "커버리지 포함 테스트 실행")
    
    if success:
        print("\n커버리지 보고서가 생성되었습니다:")
        print("- HTML 보고서: htmlcov/index.html")
        print("- JSON 보고서: coverage.json")
    
    return success, stdout, stderr


def check_test_dependencies():
    """테스트 의존성 확인"""
    required_packages = [
        'pytest',
        'pytest-cov', 
        'numpy',
        'librosa',
        'scikit-learn',
        'soundfile'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  누락된 패키지:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def generate_test_report(results):
    """테스트 결과 보고서 생성"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_suites': len(results),
            'passed_suites': sum(1 for r in results.values() if r['success']),
            'failed_suites': sum(1 for r in results.values() if not r['success'])
        }
    }
    
    # JSON 보고서 저장
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 콘솔 요약 출력
    print(f"\n{'='*60}")
    print("테스트 실행 요약")
    print(f"{'='*60}")
    print(f"총 테스트 스위트: {report['summary']['total_suites']}")
    print(f"성공: {report['summary']['passed_suites']}")
    print(f"실패: {report['summary']['failed_suites']}")
    
    for suite_name, result in results.items():
        status = "✅ 성공" if result['success'] else "❌ 실패"
        print(f"  {suite_name}: {status}")
    
    print(f"\n상세 보고서: {report_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="수박 소리 분류 시스템 테스트 실행")
    
    parser.add_argument(
        '--type', 
        choices=['unit', 'integration', 'performance', 'all'],
        default='all',
        help='실행할 테스트 유형 (기본값: all)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='코드 커버리지 포함'
    )
    
    parser.add_argument(
        '--exclude-slow',
        action='store_true', 
        help='느린 테스트 제외'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='의존성만 확인'
    )
    
    args = parser.parse_args()
    
    print("🍉 수박 소리 분류 시스템 테스트 실행기")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 의존성 확인
    if not check_test_dependencies():
        if args.check_deps:
            return 1
        print("\n의존성 문제가 있지만 테스트를 계속 진행합니다...")
    elif args.check_deps:
        print("✅ 모든 의존성이 정상적으로 설치되어 있습니다.")
        return 0
    
    # 테스트 실행
    results = {}
    
    if args.type == 'unit':
        success, stdout, stderr = run_unit_tests()
        results['unit_tests'] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
    
    elif args.type == 'integration':
        success, stdout, stderr = run_integration_tests()
        results['integration_tests'] = {
            'success': success,
            'stdout': stdout, 
            'stderr': stderr
        }
    
    elif args.type == 'performance':
        success, stdout, stderr = run_performance_tests()
        results['performance_tests'] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
    
    elif args.type == 'all':
        # 단위 테스트
        success, stdout, stderr = run_unit_tests()
        results['unit_tests'] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # 통합 테스트
        success, stdout, stderr = run_integration_tests()
        results['integration_tests'] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
        
        # 성능 테스트 (exclude_slow가 False인 경우에만)
        if not args.exclude_slow:
            success, stdout, stderr = run_performance_tests()
            results['performance_tests'] = {
                'success': success,
                'stdout': stdout,
                'stderr': stderr
            }
    
    # 커버리지 보고서 생성
    if args.coverage:
        success, stdout, stderr = run_coverage_report()
        results['coverage_report'] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        }
    
    # 보고서 생성
    report = generate_test_report(results)
    
    # 종료 코드 결정
    all_passed = all(r['success'] for r in results.values())
    
    if all_passed:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return 0
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    sys.exit(main())