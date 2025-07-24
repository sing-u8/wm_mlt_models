"""
데이터 무결성 검사 모듈

파이프라인의 각 단계에서 데이터 품질과 무결성을 검증합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ..utils.logger import LoggerMixin


@dataclass
class DataIntegrityReport:
    """데이터 무결성 검사 보고서"""
    step_name: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: List[str]
    errors: List[str]
    details: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """성공률 반환"""
        return self.passed_checks / self.total_checks if self.total_checks > 0 else 0.0


class DataIntegrityChecker(LoggerMixin):
    """
    데이터 무결성 검사기
    
    파이프라인의 각 단계에서 데이터 품질을 검증합니다.
    """
    
    def __init__(self, config=None):
        """
        무결성 검사기를 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체
        """
        self.config = config
        self.check_results = []
        
    def check_audio_features(self, X: np.ndarray, y: np.ndarray, 
                           step_name: str = "feature_extraction") -> DataIntegrityReport:
        """
        오디오 특징 데이터의 무결성을 검사합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            특징 데이터
        y : np.ndarray
            레이블 데이터
        step_name : str
            검사 단계 이름
            
        Returns:
        --------
        DataIntegrityReport
            무결성 검사 보고서
        """
        self.logger.info(f"=== {step_name} 데이터 무결성 검사 시작 ===")
        
        warnings = []
        errors = []
        checks_passed = 0
        total_checks = 0
        details = {}
        
        try:
            # 1. 기본 형태 검사
            total_checks += 1
            if X.shape[0] == len(y):
                checks_passed += 1
                self.logger.info(f"✅ 특징-레이블 길이 일치: {X.shape[0]} 샘플")
            else:
                errors.append(f"특징-레이블 길이 불일치: {X.shape[0]} vs {len(y)}")
                self.logger.error(f"❌ 특징-레이블 길이 불일치: {X.shape[0]} vs {len(y)}")
            
            # 2. 특징 벡터 차원 검사
            total_checks += 1
            expected_features = 30  # design.md 명세
            if len(X.shape) == 2 and X.shape[1] == expected_features:
                checks_passed += 1
                self.logger.info(f"✅ 특징 벡터 차원 올바름: {X.shape[1]}")
            else:
                errors.append(f"특징 벡터 차원 오류: {X.shape[1] if len(X.shape) > 1 else 'N/A'} != {expected_features}")
                self.logger.error(f"❌ 특징 벡터 차원 오류: {X.shape}")
            
            # 3. 데이터 타입 검사
            total_checks += 1
            if X.dtype in [np.float32, np.float64]:
                checks_passed += 1
                self.logger.info(f"✅ 특징 데이터 타입 올바름: {X.dtype}")
            else:
                warnings.append(f"특징 데이터 타입 주의: {X.dtype} (float 권장)")
                self.logger.warning(f"⚠️ 특징 데이터 타입: {X.dtype}")
            
            # 4. NaN/Infinity 검사
            total_checks += 1
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            
            if nan_count == 0 and inf_count == 0:
                checks_passed += 1
                self.logger.info("✅ NaN/Infinity 없음")
            else:
                if nan_count > 0:
                    errors.append(f"NaN 값 발견: {nan_count}개")
                if inf_count > 0:
                    errors.append(f"Infinity 값 발견: {inf_count}개")
                self.logger.error(f"❌ NaN: {nan_count}, Infinity: {inf_count}")
            
            # 5. 특징 값 범위 검사
            total_checks += 1
            feature_min = np.min(X)
            feature_max = np.max(X)
            feature_mean = np.mean(X)
            feature_std = np.std(X)
            
            # 일반적인 오디오 특징 범위 확인 (대략적)
            if -100 <= feature_min <= 100 and -100 <= feature_max <= 100:
                checks_passed += 1
                self.logger.info(f"✅ 특징 값 범위 정상: [{feature_min:.3f}, {feature_max:.3f}]")
            else:
                warnings.append(f"특징 값 범위 주의: [{feature_min:.3f}, {feature_max:.3f}]")
                self.logger.warning(f"⚠️ 특징 값 범위: [{feature_min:.3f}, {feature_max:.3f}]")
            
            details['feature_statistics'] = {
                'min': float(feature_min),
                'max': float(feature_max),
                'mean': float(feature_mean),
                'std': float(feature_std),
                'shape': X.shape
            }
            
            # 6. 레이블 검사
            total_checks += 1
            unique_labels = np.unique(y)
            expected_classes = len(self.config.class_names) if self.config else 3
            
            if (len(unique_labels) <= expected_classes and 
                np.all(unique_labels >= 0) and 
                np.all(unique_labels < expected_classes)):
                checks_passed += 1
                self.logger.info(f"✅ 레이블 범위 정상: {unique_labels}")
            else:
                errors.append(f"레이블 범위 오류: {unique_labels} (기대: 0-{expected_classes-1})")
                self.logger.error(f"❌ 레이블 범위 오류: {unique_labels}")
            
            # 7. 클래스 분포 검사
            total_checks += 1
            unique, counts = np.unique(y, return_counts=True)
            min_samples = np.min(counts)
            max_samples = np.max(counts)
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            if imbalance_ratio <= 10:  # 10:1 비율 이내
                checks_passed += 1
                self.logger.info(f"✅ 클래스 분포 균형: 비율 {imbalance_ratio:.2f}")
            else:
                warnings.append(f"클래스 불균형 주의: 비율 {imbalance_ratio:.2f}")
                self.logger.warning(f"⚠️ 클래스 불균형: 비율 {imbalance_ratio:.2f}")
            
            details['label_distribution'] = {
                'unique_labels': unique_labels.tolist(),
                'class_counts': dict(zip(unique.tolist(), counts.tolist())),
                'imbalance_ratio': float(imbalance_ratio)
            }
            
            # 8. 샘플 수 검사
            total_checks += 1
            n_samples = X.shape[0]
            min_samples_threshold = 10  # 최소 샘플 수
            
            if n_samples >= min_samples_threshold:
                checks_passed += 1
                self.logger.info(f"✅ 충분한 샘플 수: {n_samples}")
            else:
                warnings.append(f"샘플 수 부족: {n_samples} < {min_samples_threshold}")
                self.logger.warning(f"⚠️ 샘플 수 부족: {n_samples}")
            
            details['sample_info'] = {
                'total_samples': int(n_samples),
                'meets_minimum': n_samples >= min_samples_threshold
            }
            
        except Exception as e:
            errors.append(f"무결성 검사 중 오류: {str(e)}")
            self.logger.error(f"❌ 무결성 검사 오류: {e}")
        
        # 보고서 생성
        passed = len(errors) == 0
        failed_checks = total_checks - checks_passed
        
        report = DataIntegrityReport(
            step_name=step_name,
            passed=passed,
            total_checks=total_checks,
            passed_checks=checks_passed,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            details=details
        )
        
        # 결과 로깅
        self.logger.info(f"📊 {step_name} 무결성 검사 완료:")
        self.logger.info(f"  전체 검사: {total_checks}개")
        self.logger.info(f"  통과: {checks_passed}개")
        self.logger.info(f"  실패: {failed_checks}개")
        self.logger.info(f"  경고: {len(warnings)}개")
        self.logger.info(f"  오류: {len(errors)}개")
        self.logger.info(f"  성공률: {report.success_rate:.1%}")
        
        if passed:
            self.logger.info(f"✅ {step_name} 데이터 무결성 검사 통과")
        else:
            self.logger.error(f"❌ {step_name} 데이터 무결성 검사 실패")
            for error in errors:
                self.logger.error(f"  - {error}")
        
        if warnings:
            self.logger.warning(f"⚠️ {len(warnings)}개 경고사항:")
            for warning in warnings:
                self.logger.warning(f"  - {warning}")
        
        self.check_results.append(report)
        
        self.logger.info(f"=== {step_name} 데이터 무결성 검사 완료 ===\n")
        
        return report
    
    def check_model_outputs(self, predictions: np.ndarray, probabilities: np.ndarray = None,
                          true_labels: np.ndarray = None, 
                          step_name: str = "model_prediction") -> DataIntegrityReport:
        """
        모델 출력의 무결성을 검사합니다.
        
        Parameters:
        -----------
        predictions : np.ndarray
            예측 결과
        probabilities : np.ndarray, optional
            예측 확률
        true_labels : np.ndarray, optional
            실제 레이블
        step_name : str
            검사 단계 이름
            
        Returns:
        --------
        DataIntegrityReport
            무결성 검사 보고서
        """
        self.logger.info(f"=== {step_name} 모델 출력 무결성 검사 시작 ===")
        
        warnings = []
        errors = []
        checks_passed = 0
        total_checks = 0
        details = {}
        
        try:
            # 1. 예측 결과 기본 검사
            total_checks += 1
            if len(predictions) > 0:
                checks_passed += 1
                self.logger.info(f"✅ 예측 결과 존재: {len(predictions)}개")
            else:
                errors.append("예측 결과가 비어있음")
                self.logger.error("❌ 예측 결과가 비어있음")
            
            # 2. 예측 값 범위 검사
            total_checks += 1
            expected_classes = len(self.config.class_names) if self.config else 3
            unique_preds = np.unique(predictions)
            
            if (np.all(unique_preds >= 0) and 
                np.all(unique_preds < expected_classes)):
                checks_passed += 1
                self.logger.info(f"✅ 예측 범위 정상: {unique_preds}")
            else:
                errors.append(f"예측 범위 오류: {unique_preds} (기대: 0-{expected_classes-1})")
                self.logger.error(f"❌ 예측 범위 오류: {unique_preds}")
            
            details['prediction_info'] = {
                'total_predictions': len(predictions),
                'unique_predictions': unique_preds.tolist(),
                'prediction_distribution': dict(zip(*np.unique(predictions, return_counts=True)))
            }
            
            # 3. 확률 검사 (제공된 경우)
            if probabilities is not None:
                total_checks += 1
                if probabilities.shape[0] == len(predictions):
                    checks_passed += 1
                    self.logger.info("✅ 확률-예측 길이 일치")
                else:
                    errors.append(f"확률-예측 길이 불일치: {probabilities.shape[0]} vs {len(predictions)}")
                    self.logger.error(f"❌ 확률-예측 길이 불일치")
                
                total_checks += 1
                prob_sums = np.sum(probabilities, axis=1)
                if np.allclose(prob_sums, 1.0, atol=1e-6):
                    checks_passed += 1
                    self.logger.info("✅ 확률 합계 정상 (≈1.0)")
                else:
                    warnings.append(f"확률 합계 주의: 평균 {np.mean(prob_sums):.4f}")
                    self.logger.warning(f"⚠️ 확률 합계: 평균 {np.mean(prob_sums):.4f}")
                
                total_checks += 1
                if np.all(probabilities >= 0) and np.all(probabilities <= 1):
                    checks_passed += 1
                    self.logger.info("✅ 확률 범위 정상 [0, 1]")
                else:
                    errors.append("확률 범위 오류: [0, 1] 벗어남")
                    self.logger.error("❌ 확률 범위 오류")
                
                details['probability_info'] = {
                    'shape': probabilities.shape,
                    'mean_confidence': float(np.mean(np.max(probabilities, axis=1))),
                    'min_probability': float(np.min(probabilities)),
                    'max_probability': float(np.max(probabilities))
                }
            
            # 4. 실제 레이블과 비교 (제공된 경우)
            if true_labels is not None:
                total_checks += 1
                if len(true_labels) == len(predictions):
                    checks_passed += 1
                    accuracy = np.mean(true_labels == predictions)
                    self.logger.info(f"✅ 예측 정확도: {accuracy:.3f}")
                    
                    if accuracy < 0.1:  # 10% 미만이면 경고
                        warnings.append(f"낮은 정확도: {accuracy:.3f}")
                        self.logger.warning(f"⚠️ 낮은 정확도: {accuracy:.3f}")
                    
                    details['accuracy_info'] = {
                        'accuracy': float(accuracy),
                        'correct_predictions': int(np.sum(true_labels == predictions)),
                        'total_predictions': len(predictions)
                    }
                else:
                    errors.append(f"실제 레이블-예측 길이 불일치: {len(true_labels)} vs {len(predictions)}")
                    self.logger.error(f"❌ 실제 레이블-예측 길이 불일치")
        
        except Exception as e:
            errors.append(f"모델 출력 검사 중 오류: {str(e)}")
            self.logger.error(f"❌ 모델 출력 검사 오류: {e}")
        
        # 보고서 생성
        passed = len(errors) == 0
        failed_checks = total_checks - checks_passed
        
        report = DataIntegrityReport(
            step_name=step_name,
            passed=passed,
            total_checks=total_checks,
            passed_checks=checks_passed,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            details=details
        )
        
        # 결과 로깅
        self.logger.info(f"📊 {step_name} 출력 무결성 검사 완료:")
        self.logger.info(f"  전체 검사: {total_checks}개")
        self.logger.info(f"  통과: {checks_passed}개")
        self.logger.info(f"  실패: {failed_checks}개")
        self.logger.info(f"  성공률: {report.success_rate:.1%}")
        
        if passed:
            self.logger.info(f"✅ {step_name} 모델 출력 무결성 검사 통과")
        else:
            self.logger.error(f"❌ {step_name} 모델 출력 무결성 검사 실패")
        
        self.check_results.append(report)
        
        self.logger.info(f"=== {step_name} 모델 출력 무결성 검사 완료 ===\n")
        
        return report
    
    def check_pipeline_consistency(self, train_data: Tuple[np.ndarray, np.ndarray],
                                 val_data: Tuple[np.ndarray, np.ndarray],
                                 test_data: Tuple[np.ndarray, np.ndarray],
                                 step_name: str = "pipeline_consistency") -> DataIntegrityReport:
        """
        파이프라인 전체의 데이터 일관성을 검사합니다.
        
        Parameters:
        -----------
        train_data : Tuple
            (훈련 특징, 훈련 레이블)
        val_data : Tuple
            (검증 특징, 검증 레이블)
        test_data : Tuple
            (테스트 특징, 테스트 레이블)
        step_name : str
            검사 단계 이름
            
        Returns:
        --------
        DataIntegrityReport
            무결성 검사 보고서
        """
        self.logger.info(f"=== {step_name} 파이프라인 일관성 검사 시작 ===")
        
        warnings = []
        errors = []
        checks_passed = 0
        total_checks = 0
        details = {}
        
        try:
            X_train, y_train = train_data
            X_val, y_val = val_data
            X_test, y_test = test_data
            
            # 1. 특징 차원 일관성 검사
            total_checks += 1
            feature_dims = [X_train.shape[1], X_val.shape[1], X_test.shape[1]]
            if len(set(feature_dims)) == 1:
                checks_passed += 1
                self.logger.info(f"✅ 특징 차원 일관성: {feature_dims[0]}")
            else:
                errors.append(f"특징 차원 불일치: train={feature_dims[0]}, val={feature_dims[1]}, test={feature_dims[2]}")
                self.logger.error(f"❌ 특징 차원 불일치: {feature_dims}")
            
            # 2. 데이터 타입 일관성 검사
            total_checks += 1
            dtypes = [X_train.dtype, X_val.dtype, X_test.dtype]
            if len(set(dtypes)) == 1:
                checks_passed += 1
                self.logger.info(f"✅ 데이터 타입 일관성: {dtypes[0]}")
            else:
                warnings.append(f"데이터 타입 불일치: {dtypes}")
                self.logger.warning(f"⚠️ 데이터 타입 불일치: {dtypes}")
            
            # 3. 클래스 레이블 일관성 검사  
            total_checks += 1
            train_classes = set(np.unique(y_train))
            val_classes = set(np.unique(y_val))
            test_classes = set(np.unique(y_test))
            all_classes = train_classes | val_classes | test_classes
            
            if train_classes == val_classes == test_classes:
                checks_passed += 1
                self.logger.info(f"✅ 클래스 레이블 일관성: {sorted(all_classes)}")
            else:
                warnings.append(f"클래스 분포 차이: train={sorted(train_classes)}, val={sorted(val_classes)}, test={sorted(test_classes)}")
                self.logger.warning(f"⚠️ 클래스 분포 차이 발견")
            
            # 4. 데이터 분할 비율 검사
            total_checks += 1
            total_samples = len(X_train) + len(X_val) + len(X_test)
            train_ratio = len(X_train) / total_samples
            val_ratio = len(X_val) / total_samples
            test_ratio = len(X_test) / total_samples
            
            # 일반적인 분할 비율 (70-15-15 또는 80-10-10)
            if (0.6 <= train_ratio <= 0.9 and 
                0.05 <= val_ratio <= 0.3 and 
                0.05 <= test_ratio <= 0.3):
                checks_passed += 1
                self.logger.info(f"✅ 데이터 분할 비율 적절: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")
            else:
                warnings.append(f"데이터 분할 비율 주의: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")
                self.logger.warning(f"⚠️ 데이터 분할 비율: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")
            
            # 5. 데이터 누출 검사 (간단한 중복 검사)
            total_checks += 1
            # 훈련-검증 중복 확인 (근사치, 실제로는 해시나 더 정교한 방법 필요)
            if len(X_train) > 0 and len(X_val) > 0:
                # 간단한 평균값 비교로 대략적 중복 확인
                train_means = np.mean(X_train, axis=1)
                val_means = np.mean(X_val, axis=1)
                
                duplicates = 0
                for train_mean in train_means[:min(100, len(train_means))]:  # 처음 100개만 확인
                    if np.any(np.abs(val_means - train_mean) < 1e-6):
                        duplicates += 1
                
                if duplicates == 0:
                    checks_passed += 1
                    self.logger.info("✅ 훈련-검증 데이터 누출 없음")
                else:
                    warnings.append(f"잠재적 데이터 누출: {duplicates}개 유사 샘플")
                    self.logger.warning(f"⚠️ 잠재적 데이터 누출: {duplicates}개")
            else:
                checks_passed += 1  # 빈 데이터셋은 누출 없음으로 간주
            
            details['consistency_info'] = {
                'feature_dimensions': feature_dims,
                'data_types': [str(dt) for dt in dtypes],
                'class_distribution': {
                    'train': sorted(train_classes),
                    'val': sorted(val_classes),
                    'test': sorted(test_classes)
                },
                'split_ratios': {
                    'train': float(train_ratio),
                    'val': float(val_ratio),
                    'test': float(test_ratio)
                },
                'sample_counts': {
                    'train': len(X_train),
                    'val': len(X_val),
                    'test': len(X_test),
                    'total': total_samples
                }
            }
            
        except Exception as e:
            errors.append(f"파이프라인 일관성 검사 중 오류: {str(e)}")
            self.logger.error(f"❌ 파이프라인 일관성 검사 오류: {e}")
        
        # 보고서 생성
        passed = len(errors) == 0
        failed_checks = total_checks - checks_passed
        
        report = DataIntegrityReport(
            step_name=step_name,
            passed=passed,
            total_checks=total_checks,
            passed_checks=checks_passed,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            details=details
        )
        
        # 결과 로깅
        self.logger.info(f"📊 {step_name} 일관성 검사 완료:")
        self.logger.info(f"  전체 검사: {total_checks}개")
        self.logger.info(f"  통과: {checks_passed}개")
        self.logger.info(f"  실패: {failed_checks}개")
        self.logger.info(f"  성공률: {report.success_rate:.1%}")
        
        if passed:
            self.logger.info(f"✅ {step_name} 파이프라인 일관성 검사 통과")
        else:
            self.logger.error(f"❌ {step_name} 파이프라인 일관성 검사 실패")
        
        self.check_results.append(report)
        
        self.logger.info(f"=== {step_name} 파이프라인 일관성 검사 완료 ===\n")
        
        return report
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        전체 무결성 검사 요약 보고서를 생성합니다.
        
        Returns:
        --------
        Dict[str, Any]
            요약 보고서
        """
        if not self.check_results:
            return {'message': '실행된 무결성 검사가 없습니다.'}
        
        total_checks = sum(r.total_checks for r in self.check_results)
        total_passed = sum(r.passed_checks for r in self.check_results)
        total_failed = total_checks - total_passed
        
        total_warnings = sum(len(r.warnings) for r in self.check_results)
        total_errors = sum(len(r.errors) for r in self.check_results)
        
        overall_success_rate = total_passed / total_checks if total_checks > 0 else 0
        
        step_summaries = []
        for report in self.check_results:
            step_summaries.append({
                'step_name': report.step_name,
                'passed': report.passed,
                'success_rate': report.success_rate,
                'checks': report.total_checks,
                'warnings': len(report.warnings),
                'errors': len(report.errors)
            })
        
        summary = {
            'overall_statistics': {
                'total_checks_run': len(self.check_results),
                'total_individual_checks': total_checks,
                'total_passed_checks': total_passed,
                'total_failed_checks': total_failed,
                'overall_success_rate': overall_success_rate,
                'total_warnings': total_warnings,
                'total_errors': total_errors
            },
            'step_summaries': step_summaries,
            'overall_status': 'PASSED' if total_errors == 0 else 'FAILED'
        }
        
        return summary