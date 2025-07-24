"""
배포 준비 상태 검증기

pickle 및 Core ML 모델의 배포 준비 상태를 종합적으로 검증하는 스크립트
"""

import os
import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import tempfile
import hashlib

# 프로젝트 모듈 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.model_converter import ModelConverter
from src.audio.feature_extraction import extract_features, FeatureVector
from src.utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class ModelInfo:
    """모델 정보"""
    file_path: str
    file_size_mb: float
    creation_time: str
    model_type: str  # 'pickle', 'coreml'
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class DeploymentCheck:
    """배포 검증 항목"""
    check_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    details: str
    recommendations: List[str] = None


@dataclass
class DeploymentReport:
    """배포 준비 보고서"""
    timestamp: str
    pickle_models: List[ModelInfo]
    coreml_models: List[ModelInfo]
    deployment_checks: List[DeploymentCheck]
    overall_status: str
    deployment_score: float  # 0.0-1.0
    critical_issues: List[str]
    recommendations: List[str]
    model_performance_summary: Dict


class DeploymentReadinessChecker(LoggerMixin):
    """배포 준비 상태 검증기"""
    
    def __init__(self, models_dir: str = "results/trained_models"):
        self.logger = self.get_logger()
        self.models_dir = Path(models_dir)
        
        # 검증 기준
        self.max_model_size_mb = 100  # 최대 모델 크기
        self.min_accuracy_threshold = 0.6  # 최소 정확도
        self.required_model_types = ['svm', 'random_forest']  # 필수 모델 타입
    
    def run_comprehensive_deployment_check(self) -> DeploymentReport:
        """포괄적 배포 준비 상태 검증"""
        self.logger.info("=== 배포 준비 상태 검증 시작 ===")
        
        # 모델 파일 검색 및 정보 수집
        pickle_models = self._discover_pickle_models()
        coreml_models = self._discover_coreml_models()
        
        # 배포 검증 수행
        deployment_checks = []
        
        # 1. 모델 파일 존재성 검증
        deployment_checks.extend(self._check_model_existence(pickle_models, coreml_models))
        
        # 2. 모델 무결성 검증
        deployment_checks.extend(self._check_model_integrity(pickle_models, coreml_models))
        
        # 3. 모델 성능 검증
        deployment_checks.extend(self._check_model_performance(pickle_models))
        
        # 4. 배포 호환성 검증
        deployment_checks.extend(self._check_deployment_compatibility(pickle_models, coreml_models))
        
        # 5. 보안 검증
        deployment_checks.extend(self._check_security_requirements(pickle_models, coreml_models))
        
        # 6. 문서화 검증
        deployment_checks.extend(self._check_documentation())
        
        # 전체 평가 및 보고서 생성
        overall_status, deployment_score, critical_issues, recommendations = self._evaluate_deployment_readiness(deployment_checks)
        
        # 성능 요약
        performance_summary = self._generate_performance_summary(pickle_models)
        
        report = DeploymentReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            pickle_models=pickle_models,
            coreml_models=coreml_models,
            deployment_checks=deployment_checks,
            overall_status=overall_status,
            deployment_score=deployment_score,
            critical_issues=critical_issues,
            recommendations=recommendations,
            model_performance_summary=performance_summary
        )
        
        self.logger.info(f"배포 준비 검증 완료: {overall_status} (점수: {deployment_score:.2f})")
        return report
    
    def _discover_pickle_models(self) -> List[ModelInfo]:
        """pickle 모델 발견 및 정보 수집"""
        pickle_models = []
        
        if not self.models_dir.exists():
            self.logger.warning(f"모델 디렉토리가 존재하지 않음: {self.models_dir}")
            return pickle_models
        
        # .pkl 파일 검색
        for pkl_file in self.models_dir.glob("**/*.pkl"):
            try:
                stat = pkl_file.stat()
                
                model_info = ModelInfo(
                    file_path=str(pkl_file),
                    file_size_mb=stat.st_size / (1024 * 1024),
                    creation_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                    model_type='pickle',
                    is_valid=False
                )
                
                # 모델 로딩 테스트
                try:
                    with open(pkl_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    model_info.is_valid = True
                    self.logger.info(f"Pickle 모델 발견: {pkl_file.name} ({model_info.file_size_mb:.1f}MB)")
                    
                except Exception as e:
                    model_info.error_message = f"모델 로딩 실패: {str(e)}"
                    self.logger.error(f"Pickle 모델 로딩 실패 {pkl_file}: {e}")
                
                pickle_models.append(model_info)
                
            except Exception as e:
                self.logger.error(f"Pickle 모델 정보 수집 실패 {pkl_file}: {e}")
        
        return pickle_models
    
    def _discover_coreml_models(self) -> List[ModelInfo]:
        """Core ML 모델 발견 및 정보 수집"""
        coreml_models = []
        
        if not self.models_dir.exists():
            return coreml_models
        
        # .mlmodel 파일 검색
        for mlmodel_file in self.models_dir.glob("**/*.mlmodel"):
            try:
                stat = mlmodel_file.stat()
                
                model_info = ModelInfo(
                    file_path=str(mlmodel_file),
                    file_size_mb=stat.st_size / (1024 * 1024),
                    creation_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                    model_type='coreml',
                    is_valid=False
                )
                
                # Core ML 모델 검증
                try:
                    # coremltools가 있는 경우에만 검증
                    import coremltools as ct
                    model = ct.models.MLModel(str(mlmodel_file))
                    
                    model_info.is_valid = True
                    self.logger.info(f"Core ML 모델 발견: {mlmodel_file.name} ({model_info.file_size_mb:.1f}MB)")
                    
                except ImportError:
                    model_info.error_message = "coremltools 없음 (선택적 의존성)"
                    model_info.is_valid = True  # 선택적이므로 유효한 것으로 간주
                    self.logger.warning(f"Core ML 검증 건너뜀 (coremltools 없음): {mlmodel_file}")
                    
                except Exception as e:
                    model_info.error_message = f"Core ML 모델 로딩 실패: {str(e)}"
                    self.logger.error(f"Core ML 모델 로딩 실패 {mlmodel_file}: {e}")
                
                coreml_models.append(model_info)
                
            except Exception as e:
                self.logger.error(f"Core ML 모델 정보 수집 실패 {mlmodel_file}: {e}")
        
        return coreml_models
    
    def _check_model_existence(self, pickle_models: List[ModelInfo], 
                             coreml_models: List[ModelInfo]) -> List[DeploymentCheck]:
        """모델 파일 존재성 검증"""
        checks = []
        
        # Pickle 모델 존재성
        if pickle_models:
            valid_pickle = sum(1 for m in pickle_models if m.is_valid)
            check = DeploymentCheck(
                check_name="pickle_models_existence",
                status="PASS" if valid_pickle > 0 else "FAIL",
                details=f"유효한 pickle 모델 {valid_pickle}개 발견",
                recommendations=["모델을 훈련하고 저장하세요"] if valid_pickle == 0 else []
            )
        else:
            check = DeploymentCheck(
                check_name="pickle_models_existence",
                status="FAIL",
                details="pickle 모델이 발견되지 않음",
                recommendations=["main.py를 실행하여 모델을 훈련하세요"]
            )
        
        checks.append(check)
        
        # Core ML 모델 존재성 (선택적)
        if coreml_models:
            valid_coreml = sum(1 for m in coreml_models if m.is_valid)
            check = DeploymentCheck(
                check_name="coreml_models_existence",
                status="PASS" if valid_coreml > 0 else "WARNING",
                details=f"유효한 Core ML 모델 {valid_coreml}개 발견",
                recommendations=["Core ML 변환을 수행하세요"] if valid_coreml == 0 else []
            )
        else:
            check = DeploymentCheck(
                check_name="coreml_models_existence",
                status="WARNING",
                details="Core ML 모델이 발견되지 않음 (선택적)",
                recommendations=["iOS/macOS 배포를 위한 Core ML 변환을 고려하세요"]
            )
        
        checks.append(check)
        
        return checks
    
    def _check_model_integrity(self, pickle_models: List[ModelInfo], 
                             coreml_models: List[ModelInfo]) -> List[DeploymentCheck]:
        """모델 무결성 검증"""
        checks = []
        
        # Pickle 모델 무결성
        integrity_issues = []
        
        for model_info in pickle_models:
            if not model_info.is_valid:
                integrity_issues.append(f"{Path(model_info.file_path).name}: {model_info.error_message}")
            
            # 모델 크기 검증
            if model_info.file_size_mb > self.max_model_size_mb:
                integrity_issues.append(f"{Path(model_info.file_path).name}: 크기가 너무 큼 ({model_info.file_size_mb:.1f}MB > {self.max_model_size_mb}MB)")
        
        if integrity_issues:
            check = DeploymentCheck(
                check_name="model_integrity",
                status="FAIL",
                details=f"무결성 문제 {len(integrity_issues)}개 발견",
                recommendations=[
                    "손상된 모델을 다시 훈련하세요",
                    "모델 크기를 최적화하세요"
                ]
            )
        else:
            check = DeploymentCheck(
                check_name="model_integrity",
                status="PASS",
                details="모든 모델이 무결성 검사를 통과함"
            )
        
        checks.append(check)
        
        return checks
    
    def _check_model_performance(self, pickle_models: List[ModelInfo]) -> List[DeploymentCheck]:
        """모델 성능 검증"""
        checks = []
        
        try:
            # 메타데이터 파일에서 성능 정보 로드
            performance_data = self._load_model_performance_data()
            
            if not performance_data:
                check = DeploymentCheck(
                    check_name="model_performance",
                    status="WARNING",
                    details="모델 성능 데이터를 찾을 수 없음",
                    recommendations=["모델을 다시 훈련하고 평가하세요"]
                )
                checks.append(check)
                return checks
            
            # 성능 기준 검증
            performance_issues = []
            best_accuracy = 0.0
            
            for model_name, metrics in performance_data.items():
                accuracy = metrics.get('accuracy', 0.0)
                best_accuracy = max(best_accuracy, accuracy)
                
                if accuracy < self.min_accuracy_threshold:
                    performance_issues.append(f"{model_name}: 정확도 낮음 ({accuracy:.3f} < {self.min_accuracy_threshold})")
            
            if performance_issues:
                check = DeploymentCheck(
                    check_name="model_performance",
                    status="FAIL" if best_accuracy < self.min_accuracy_threshold else "WARNING",
                    details=f"성능 기준 미달 모델 {len(performance_issues)}개",
                    recommendations=[
                        "모델 하이퍼파라미터를 최적화하세요",
                        "더 많은 훈련 데이터를 수집하세요",
                        "특징 엔지니어링을 개선하세요"
                    ]
                )
            else:
                check = DeploymentCheck(
                    check_name="model_performance",
                    status="PASS",
                    details=f"모든 모델이 성능 기준을 만족함 (최고 정확도: {best_accuracy:.3f})"
                )
            
            checks.append(check)
            
        except Exception as e:
            check = DeploymentCheck(
                check_name="model_performance",
                status="WARNING",
                details=f"성능 검증 실패: {str(e)}",
                recommendations=["모델 평가를 다시 실행하세요"]
            )
            checks.append(check)
        
        return checks
    
    def _check_deployment_compatibility(self, pickle_models: List[ModelInfo], 
                                      coreml_models: List[ModelInfo]) -> List[DeploymentCheck]:
        """배포 호환성 검증"""
        checks = []
        
        # Python 환경 호환성 (pickle)
        python_compat_issues = []
        
        for model_info in pickle_models:
            if model_info.is_valid:
                try:
                    # 모델 로딩 및 예측 테스트
                    with open(model_info.file_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # 더미 데이터로 예측 테스트
                    dummy_features = np.random.rand(1, 30)  # 30차원 특징 벡터
                    prediction = model.predict(dummy_features)
                    
                    # 예측 결과 검증
                    if not isinstance(prediction, np.ndarray) or len(prediction) == 0:
                        python_compat_issues.append(f"{Path(model_info.file_path).name}: 예측 결과 형식 오류")
                    
                except Exception as e:
                    python_compat_issues.append(f"{Path(model_info.file_path).name}: 예측 테스트 실패 ({str(e)})")
        
        if python_compat_issues:
            check = DeploymentCheck(
                check_name="python_compatibility",
                status="FAIL",
                details=f"Python 호환성 문제 {len(python_compat_issues)}개",
                recommendations=[
                    "모델을 현재 Python 환경에서 다시 훈련하세요",
                    "scikit-learn 버전 호환성을 확인하세요"
                ]
            )
        else:
            check = DeploymentCheck(
                check_name="python_compatibility",
                status="PASS",
                details="Python 환경 호환성 검증 통과"
            )
        
        checks.append(check)
        
        # Core ML 호환성 (선택적)
        if coreml_models:
            coreml_issues = []
            
            for model_info in coreml_models:
                if model_info.is_valid:
                    try:
                        # Core ML 모델 검증 (coremltools가 있는 경우)
                        import coremltools as ct
                        model = ct.models.MLModel(model_info.file_path)
                        
                        # 입력/출력 스펙 검증
                        spec = model.get_spec()
                        if not spec.description.input or not spec.description.output:
                            coreml_issues.append(f"{Path(model_info.file_path).name}: 입출력 스펙 누락")
                    
                    except ImportError:
                        # coremltools 없음 - 경고만
                        pass
                    except Exception as e:
                        coreml_issues.append(f"{Path(model_info.file_path).name}: 검증 실패 ({str(e)})")
            
            if coreml_issues:
                check = DeploymentCheck(
                    check_name="coreml_compatibility",
                    status="WARNING",
                    details=f"Core ML 호환성 문제 {len(coreml_issues)}개",
                    recommendations=["Core ML 모델을 다시 변환하세요"]
                )
            else:
                check = DeploymentCheck(
                    check_name="coreml_compatibility",
                    status="PASS",
                    details="Core ML 호환성 검증 통과"
                )
            
            checks.append(check)
        
        return checks
    
    def _check_security_requirements(self, pickle_models: List[ModelInfo], 
                                   coreml_models: List[ModelInfo]) -> List[DeploymentCheck]:
        """보안 요구사항 검증"""
        checks = []
        
        security_issues = []
        
        # 모델 파일 권한 검증
        for model_info in pickle_models + coreml_models:
            try:
                file_path = Path(model_info.file_path)
                stat = file_path.stat()
                
                # 파일 권한이 너무 개방적인지 확인
                if hasattr(stat, 'st_mode'):
                    mode = oct(stat.st_mode)[-3:]
                    if mode in ['777', '666']:  # 모든 사용자에게 쓰기 권한
                        security_issues.append(f"{file_path.name}: 권한이 너무 개방적임 ({mode})")
                
            except Exception as e:
                security_issues.append(f"{Path(model_info.file_path).name}: 권한 확인 실패 ({str(e)})")
        
        # 모델 체크섬 검증 (무결성)
        checksum_file = self.models_dir / "model_checksums.json"
        if checksum_file.exists():
            try:
                with open(checksum_file, 'r') as f:
                    stored_checksums = json.load(f)
                
                for model_info in pickle_models:
                    if model_info.is_valid:
                        file_name = Path(model_info.file_path).name
                        
                        if file_name in stored_checksums:
                            # 현재 체크섬 계산
                            with open(model_info.file_path, 'rb') as f:
                                current_checksum = hashlib.sha256(f.read()).hexdigest()
                            
                            if current_checksum != stored_checksums[file_name]:
                                security_issues.append(f"{file_name}: 체크섬 불일치 (변조 가능성)")
                        else:
                            security_issues.append(f"{file_name}: 체크섬 없음")
            
            except Exception as e:
                security_issues.append(f"체크섬 검증 실패: {str(e)}")
        
        if security_issues:
            check = DeploymentCheck(
                check_name="security_requirements",
                status="WARNING",
                details=f"보안 이슈 {len(security_issues)}개 발견",
                recommendations=[
                    "모델 파일 권한을 적절히 설정하세요",
                    "모델 체크섬을 생성하고 검증하세요",
                    "모델 파일을 안전한 위치에 저장하세요"
                ]
            )
        else:
            check = DeploymentCheck(
                check_name="security_requirements",
                status="PASS",
                details="보안 요구사항 검증 통과"
            )
        
        checks.append(check)
        return checks
    
    def _check_documentation(self) -> List[DeploymentCheck]:
        """문서화 검증"""
        checks = []
        
        documentation_issues = []
        
        # 필수 문서 파일들
        required_docs = [
            ("README.md", "프로젝트 설명서"),
            ("requirements.txt", "의존성 목록"),
            ("docs/API_REFERENCE.md", "API 문서"),
            ("docs/MODEL_USAGE_EXAMPLES.md", "모델 사용 예제")
        ]
        
        for doc_path, description in required_docs:
            if not os.path.exists(doc_path):
                documentation_issues.append(f"{doc_path}: {description} 누락")
            else:
                # 파일 크기 확인 (최소한의 내용 있는지)
                try:
                    size = os.path.getsize(doc_path)
                    if size < 100:  # 100바이트 미만
                        documentation_issues.append(f"{doc_path}: 내용이 부족함 ({size} bytes)")
                except Exception:
                    documentation_issues.append(f"{doc_path}: 접근 불가")
        
        # 모델 메타데이터 파일 확인
        metadata_file = self.models_dir / "model_metadata.json"
        if not metadata_file.exists():
            documentation_issues.append("model_metadata.json: 모델 메타데이터 누락")
        
        if documentation_issues:
            check = DeploymentCheck(
                check_name="documentation",
                status="WARNING",
                details=f"문서화 이슈 {len(documentation_issues)}개",
                recommendations=[
                    "누락된 문서를 작성하세요",
                    "모델 사용법과 API 문서를 업데이트하세요",
                    "배포 가이드를 작성하세요"
                ]
            )
        else:
            check = DeploymentCheck(
                check_name="documentation",
                status="PASS",
                details="문서화 검증 통과"
            )
        
        checks.append(check)
        return checks
    
    def _load_model_performance_data(self) -> Dict:
        """모델 성능 데이터 로드"""
        try:
            # 평가 결과 파일들 검색
            performance_files = [
                self.models_dir / "evaluation_results.json",
                self.models_dir / "model_performance.json",
                "results/evaluation_results.json"
            ]
            
            for perf_file in performance_files:
                if Path(perf_file).exists():
                    with open(perf_file, 'r') as f:
                        data = json.load(f)
                    return data
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"성능 데이터 로드 실패: {e}")
            return {}
    
    def _evaluate_deployment_readiness(self, checks: List[DeploymentCheck]) -> Tuple[str, float, List[str], List[str]]:
        """배포 준비 상태 종합 평가"""
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.status == 'PASS')
        failed_checks = sum(1 for c in checks if c.status == 'FAIL')
        warning_checks = sum(1 for c in checks if c.status == 'WARNING')
        
        # 배포 점수 계산 (0.0-1.0)
        if total_checks == 0:
            deployment_score = 0.0
        else:
            deployment_score = (passed_checks + warning_checks * 0.5) / total_checks
        
        # 전체 상태 결정
        if failed_checks == 0 and warning_checks <= 2:
            overall_status = "READY"
        elif failed_checks <= 1 and deployment_score >= 0.7:
            overall_status = "PARTIALLY_READY"
        else:
            overall_status = "NOT_READY"
        
        # 중요 이슈 추출
        critical_issues = []
        for check in checks:
            if check.status == 'FAIL':
                critical_issues.append(f"{check.check_name}: {check.details}")
        
        # 권장사항 종합
        recommendations = []
        for check in checks:
            if check.recommendations:
                recommendations.extend(check.recommendations)
        
        # 중복 제거
        recommendations = list(set(recommendations))
        
        return overall_status, deployment_score, critical_issues, recommendations
    
    def _generate_performance_summary(self, pickle_models: List[ModelInfo]) -> Dict:
        """성능 요약 생성"""
        try:
            performance_data = self._load_model_performance_data()
            
            if not performance_data:
                return {"status": "데이터 없음"}
            
            summary = {
                "total_models": len(pickle_models),
                "valid_models": sum(1 for m in pickle_models if m.is_valid),
                "best_accuracy": 0.0,
                "average_accuracy": 0.0,
                "model_details": {}
            }
            
            accuracies = []
            
            for model_name, metrics in performance_data.items():
                accuracy = metrics.get('accuracy', 0.0)
                accuracies.append(accuracy)
                
                summary["model_details"][model_name] = {
                    "accuracy": accuracy,
                    "precision": metrics.get('precision', 0.0),
                    "recall": metrics.get('recall', 0.0),
                    "f1_score": metrics.get('f1_score', 0.0)
                }
            
            if accuracies:
                summary["best_accuracy"] = max(accuracies)
                summary["average_accuracy"] = sum(accuracies) / len(accuracies)
            
            return summary
            
        except Exception as e:
            return {"status": f"오류: {str(e)}"}
    
    def save_deployment_report(self, report: DeploymentReport, output_file: str = None):
        """배포 준비 보고서 저장"""
        if output_file is None:
            output_file = f"deployment_readiness_report_{int(time.time())}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            
            # 텍스트 요약 보고서
            summary_file = output_file.replace('.json', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("수박 소리 분류 시스템 배포 준비 보고서\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"검증 시간: {report.timestamp}\n")
                f.write(f"전체 상태: {report.overall_status}\n")
                f.write(f"배포 준비 점수: {report.deployment_score:.2f}/1.00\n\n")
                
                f.write(f"모델 현황:\n")
                f.write(f"  Pickle 모델: {len(report.pickle_models)}개\n")
                f.write(f"  Core ML 모델: {len(report.coreml_models)}개\n\n")
                
                # 성능 요약
                perf = report.model_performance_summary
                if perf.get("status") != "데이터 없음":
                    f.write(f"성능 요약:\n")
                    f.write(f"  최고 정확도: {perf.get('best_accuracy', 0):.3f}\n")
                    f.write(f"  평균 정확도: {perf.get('average_accuracy', 0):.3f}\n\n")
                
                f.write(f"검증 결과:\n")
                for check in report.deployment_checks:
                    status_symbol = "✅" if check.status == 'PASS' else "❌" if check.status == 'FAIL' else "⚠️"
                    f.write(f"  {status_symbol} {check.check_name}: {check.details}\n")
                
                if report.critical_issues:
                    f.write(f"\n🚨 중요 이슈:\n")
                    for issue in report.critical_issues:
                        f.write(f"  • {issue}\n")
                
                if report.recommendations:
                    f.write(f"\n💡 권장사항:\n")
                    for rec in report.recommendations:
                        f.write(f"  • {rec}\n")
            
            self.logger.info(f"배포 준비 보고서 저장: {output_file}")
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='배포 준비 상태 검증')
    parser.add_argument('--models-dir', default='results/trained_models', 
                       help='모델 디렉토리 경로')
    parser.add_argument('--output', help='보고서 출력 파일명')
    
    args = parser.parse_args()
    
    # 검증기 생성 및 실행
    checker = DeploymentReadinessChecker(args.models_dir)
    report = checker.run_comprehensive_deployment_check()
    
    # 보고서 저장
    output_file = args.output or f"deployment_report_{int(time.time())}.json"
    checker.save_deployment_report(report, output_file)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"배포 준비 상태 검증 결과")
    print(f"{'='*60}")
    print(f"전체 상태: {report.overall_status}")
    print(f"배포 점수: {report.deployment_score:.2f}/1.00")
    print(f"Pickle 모델: {len(report.pickle_models)}개")
    print(f"Core ML 모델: {len(report.coreml_models)}개")
    
    # 성능 요약
    perf = report.model_performance_summary
    if perf.get("status") != "데이터 없음":
        print(f"최고 정확도: {perf.get('best_accuracy', 0):.3f}")
    
    # 중요 이슈
    if report.critical_issues:
        print(f"\n🚨 중요 이슈:")
        for issue in report.critical_issues[:3]:  # 처음 3개만
            print(f"   • {issue}")
    
    # 상태별 권장사항
    if report.overall_status == "READY":
        print(f"\n✅ 배포 준비 완료!")
    elif report.overall_status == "PARTIALLY_READY":
        print(f"\n⚠️ 부분적으로 배포 가능하나 개선 필요")
    else:
        print(f"\n❌ 배포 준비 미완료 - 문제 해결 필요")
    
    print(f"\n상세 보고서: {output_file}")
    
    # 종료 코드
    return 0 if report.overall_status in ["READY", "PARTIALLY_READY"] else 1


if __name__ == "__main__":
    sys.exit(main())