"""
모델 평가 및 메트릭 모듈

이 모듈은 훈련된 머신러닝 모델의 성능을 종합적으로 평가하고
다양한 메트릭과 시각화를 제공합니다.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import cross_val_score

from ..utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class ClassificationMetrics:
    """
    분류 성능 메트릭을 담는 데이터 클래스.
    """
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    
    # 클래스별 메트릭
    class_precision: Dict[str, float]
    class_recall: Dict[str, float]
    class_f1: Dict[str, float]
    class_support: Dict[str, int]
    
    # 혼동 행렬
    confusion_matrix: np.ndarray
    
    # ROC/PR 곡선 (다중 클래스의 경우 macro average)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


@dataclass
class ModelComparison:
    """
    모델 비교 결과를 담는 데이터 클래스.
    """
    model1_name: str
    model2_name: str
    model1_metrics: ClassificationMetrics
    model2_metrics: ClassificationMetrics
    
    # 통계적 유의성 테스트
    accuracy_ttest: Dict[str, float]  # {'statistic': float, 'pvalue': float}
    f1_ttest: Dict[str, float]
    
    # 성능 차이
    accuracy_diff: float
    f1_macro_diff: float
    
    # 승자
    better_model: str
    significance_level: float = 0.05


@dataclass
class EvaluationReport:
    """
    전체 평가 보고서를 담는 데이터 클래스.
    """
    evaluation_id: str
    created_at: str
    dataset_info: Dict[str, Any]
    
    # 개별 모델 성능
    model_metrics: Dict[str, ClassificationMetrics]
    
    # 모델 비교
    model_comparison: Optional[ModelComparison] = None
    
    # 추가 분석
    feature_importance: Optional[Dict[str, List[Tuple[str, float]]]] = None
    cross_validation_scores: Optional[Dict[str, List[float]]] = None
    
    # 메타데이터
    config_info: Dict[str, Any] = None


class ModelEvaluator(LoggerMixin):
    """
    모델 평가를 위한 클래스.
    
    design.md에 명시된 인터페이스를 구현합니다.
    """
    
    def __init__(self, config=None):
        """
        평가 구성으로 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        self.evaluation_results = {}
        
        # 시각화 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info("ModelEvaluator 초기화됨")
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str, class_names: List[str] = None) -> ClassificationMetrics:
        """
        단일 모델을 평가합니다.
        
        Parameters:
        -----------
        model : sklearn model
            평가할 훈련된 모델
        X_test : np.ndarray
            테스트 특징 데이터
        y_test : np.ndarray
            테스트 라벨 데이터
        model_name : str
            모델 이름
        class_names : List[str], optional
            클래스 이름 목록
            
        Returns:
        --------
        ClassificationMetrics
            평가 메트릭
        """
        if class_names is None:
            class_names = self.config.class_names
        
        self.logger.info(f"=== {model_name} 모델 평가 시작 ===")
        self.logger.info(f"테스트 데이터: {X_test.shape[0]}개 샘플, {X_test.shape[1]}개 특징")
        
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 기본 메트릭 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        # 클래스별 상세 메트릭
        class_report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 클래스별 메트릭 추출
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        class_support = {}
        
        for i, class_name in enumerate(class_names):
            if class_name in class_report:
                class_precision[class_name] = class_report[class_name]['precision']
                class_recall[class_name] = class_report[class_name]['recall']
                class_f1[class_name] = class_report[class_name]['f1-score']
                class_support[class_name] = class_report[class_name]['support']
        
        # ROC AUC 계산 (확률 예측이 가능한 경우)
        roc_auc = None
        pr_auc = None
        
        try:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                
                if len(class_names) == 2:
                    # 이진 분류의 경우
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob[:, 1])
                    pr_auc = auc(recall_curve, precision_curve)
                else:
                    # 다중 클래스의 경우 macro average
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                    
        except Exception as e:
            self.logger.warning(f"ROC/PR AUC 계산 실패: {e}")
        
        # 메트릭 객체 생성
        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            precision_weighted=precision_weighted,
            recall_macro=recall_macro,
            recall_weighted=recall_weighted,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            class_precision=class_precision,
            class_recall=class_recall,
            class_f1=class_f1,
            class_support=class_support,
            confusion_matrix=cm,
            roc_auc=roc_auc,
            pr_auc=pr_auc
        )
        
        # 결과 로깅
        self.logger.info(f"=== {model_name} 평가 결과 ===")
        self.logger.info(f"정확도: {accuracy:.4f}")
        self.logger.info(f"정밀도 (macro): {precision_macro:.4f}")
        self.logger.info(f"재현율 (macro): {recall_macro:.4f}")
        self.logger.info(f"F1-score (macro): {f1_macro:.4f}")
        
        if roc_auc is not None:
            self.logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # 클래스별 성능
        self.logger.info("클래스별 성능:")
        for class_name in class_names:
            if class_name in class_precision:
                self.logger.info(f"  {class_name}: P={class_precision[class_name]:.3f}, "
                               f"R={class_recall[class_name]:.3f}, "
                               f"F1={class_f1[class_name]:.3f}")
        
        return metrics
    
    def compare_models(self, model1, model2, X_test: np.ndarray, y_test: np.ndarray,
                      model1_name: str, model2_name: str, 
                      cv_folds: int = 5) -> ModelComparison:
        """
        두 모델을 비교합니다.
        
        Parameters:
        -----------
        model1, model2 : sklearn models
            비교할 두 모델
        X_test : np.ndarray
            테스트 특징 데이터
        y_test : np.ndarray
            테스트 라벨 데이터
        model1_name, model2_name : str
            모델 이름들
        cv_folds : int
            교차 검증 폴드 수
            
        Returns:
        --------
        ModelComparison
            모델 비교 결과
        """
        self.logger.info(f"=== 모델 비교: {model1_name} vs {model2_name} ===")
        
        # 개별 모델 평가
        metrics1 = self.evaluate_model(model1, X_test, y_test, model1_name)
        metrics2 = self.evaluate_model(model2, X_test, y_test, model2_name)
        
        # 교차 검증을 통한 통계적 유의성 테스트
        # 전체 데이터셋이 필요하므로 여기서는 테스트 세트만 사용
        try:
            # 정확도 비교
            cv_scores1_acc = cross_val_score(model1, X_test, y_test, cv=cv_folds, scoring='accuracy')
            cv_scores2_acc = cross_val_score(model2, X_test, y_test, cv=cv_folds, scoring='accuracy')
            
            acc_ttest = stats.ttest_ind(cv_scores1_acc, cv_scores2_acc)
            accuracy_ttest = {'statistic': acc_ttest.statistic, 'pvalue': acc_ttest.pvalue}
            
            # F1-score 비교
            cv_scores1_f1 = cross_val_score(model1, X_test, y_test, cv=cv_folds, scoring='f1_macro')
            cv_scores2_f1 = cross_val_score(model2, X_test, y_test, cv=cv_folds, scoring='f1_macro')
            
            f1_ttest = stats.ttest_ind(cv_scores1_f1, cv_scores2_f1)
            f1_ttest_result = {'statistic': f1_ttest.statistic, 'pvalue': f1_ttest.pvalue}
            
        except Exception as e:
            self.logger.warning(f"통계적 유의성 테스트 실패: {e}")
            accuracy_ttest = {'statistic': 0.0, 'pvalue': 1.0}
            f1_ttest_result = {'statistic': 0.0, 'pvalue': 1.0}
        
        # 성능 차이 계산
        accuracy_diff = metrics1.accuracy - metrics2.accuracy
        f1_macro_diff = metrics1.f1_macro - metrics2.f1_macro
        
        # 더 좋은 모델 결정
        if abs(accuracy_diff) > abs(f1_macro_diff):
            better_model = model1_name if accuracy_diff > 0 else model2_name
        else:
            better_model = model1_name if f1_macro_diff > 0 else model2_name
        
        comparison = ModelComparison(
            model1_name=model1_name,
            model2_name=model2_name,
            model1_metrics=metrics1,
            model2_metrics=metrics2,
            accuracy_ttest=accuracy_ttest,
            f1_ttest=f1_ttest_result,
            accuracy_diff=accuracy_diff,
            f1_macro_diff=f1_macro_diff,
            better_model=better_model
        )
        
        # 비교 결과 로깅
        self.logger.info(f"=== 모델 비교 결과 ===")
        self.logger.info(f"{model1_name} vs {model2_name}:")
        self.logger.info(f"  정확도: {metrics1.accuracy:.4f} vs {metrics2.accuracy:.4f} (차이: {accuracy_diff:+.4f})")
        self.logger.info(f"  F1 (macro): {metrics1.f1_macro:.4f} vs {metrics2.f1_macro:.4f} (차이: {f1_macro_diff:+.4f})")
        self.logger.info(f"  더 좋은 모델: {better_model}")
        
        if accuracy_ttest['pvalue'] < 0.05:
            self.logger.info(f"  정확도 차이가 통계적으로 유의함 (p={accuracy_ttest['pvalue']:.4f})")
        else:
            self.logger.info(f"  정확도 차이가 통계적으로 유의하지 않음 (p={accuracy_ttest['pvalue']:.4f})")
        
        return comparison
    
    def plot_confusion_matrix(self, metrics: ClassificationMetrics, model_name: str,
                             class_names: List[str] = None, save_path: str = None) -> str:
        """
        혼동 행렬을 시각화합니다.
        
        Parameters:
        -----------
        metrics : ClassificationMetrics
            평가 메트릭
        model_name : str
            모델 이름
        class_names : List[str], optional
            클래스 이름 목록
        save_path : str, optional
            저장 경로
            
        Returns:
        --------
        str
            저장된 파일 경로
        """
        if class_names is None:
            class_names = self.config.class_names
        
        plt.figure(figsize=(8, 6))
        
        # 정규화된 혼동 행렬
        cm_normalized = metrics.confusion_matrix.astype('float') / metrics.confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {metrics.accuracy:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.model_output_dir, "visualizations")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"{model_name}_confusion_matrix.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"혼동 행렬 저장됨: {save_path}")
        return save_path
    
    def plot_model_comparison(self, comparison: ModelComparison, save_path: str = None) -> str:
        """
        모델 비교 결과를 시각화합니다.
        
        Parameters:
        -----------
        comparison : ModelComparison
            모델 비교 결과
        save_path : str, optional
            저장 경로
            
        Returns:
        --------
        str
            저장된 파일 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 주요 메트릭 비교
        metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
        model1_values = [
            comparison.model1_metrics.accuracy,
            comparison.model1_metrics.precision_macro,
            comparison.model1_metrics.recall_macro,
            comparison.model1_metrics.f1_macro
        ]
        model2_values = [
            comparison.model2_metrics.accuracy,
            comparison.model2_metrics.precision_macro,
            comparison.model2_metrics.recall_macro,
            comparison.model2_metrics.f1_macro
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, model1_values, width, label=comparison.model1_name, alpha=0.8)
        axes[0, 0].bar(x + width/2, model2_values, width, label=comparison.model2_name, alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 클래스별 F1-score 비교
        class_names = list(comparison.model1_metrics.class_f1.keys())
        model1_class_f1 = [comparison.model1_metrics.class_f1[cls] for cls in class_names]
        model2_class_f1 = [comparison.model2_metrics.class_f1[cls] for cls in class_names]
        
        x_class = np.arange(len(class_names))
        axes[0, 1].bar(x_class - width/2, model1_class_f1, width, label=comparison.model1_name, alpha=0.8)
        axes[0, 1].bar(x_class + width/2, model2_class_f1, width, label=comparison.model2_name, alpha=0.8)
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('Class-wise F1-Score Comparison')
        axes[0, 1].set_xticks(x_class)
        axes[0, 1].set_xticklabels(class_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 혼동 행렬 1
        sns.heatmap(comparison.model1_metrics.confusion_matrix, 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1, 0])
        axes[1, 0].set_title(f'{comparison.model1_name} - Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # 4. 혼동 행렬 2
        sns.heatmap(comparison.model2_metrics.confusion_matrix, 
                   annot=True, fmt='d', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1, 1])
        axes[1, 1].set_title(f'{comparison.model2_name} - Confusion Matrix')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.model_output_dir, "visualizations")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"model_comparison_{comparison.model1_name}_vs_{comparison.model2_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"모델 비교 시각화 저장됨: {save_path}")
        return save_path
    
    def generate_evaluation_report(self, models: Dict[str, Any], X_test: np.ndarray, 
                                 y_test: np.ndarray, dataset_info: Dict[str, Any] = None,
                                 save_report: bool = True) -> EvaluationReport:
        """
        종합 평가 보고서를 생성합니다.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            모델명: 모델 객체 딕셔너리
        X_test : np.ndarray
            테스트 특징 데이터
        y_test : np.ndarray
            테스트 라벨 데이터
        dataset_info : Dict[str, Any], optional
            데이터셋 정보
        save_report : bool
            보고서 저장 여부
            
        Returns:
        --------
        EvaluationReport
            종합 평가 보고서
        """
        self.logger.info("=== 종합 평가 보고서 생성 시작 ===")
        
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 개별 모델 평가
        model_metrics = {}
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            model_metrics[model_name] = metrics
            
            # 혼동 행렬 시각화
            self.plot_confusion_matrix(metrics, model_name)
        
        # 모델 비교 (2개 모델인 경우)
        model_comparison = None
        if len(models) == 2:
            model_names = list(models.keys())
            model_comparison = self.compare_models(
                models[model_names[0]], models[model_names[1]],
                X_test, y_test,
                model_names[0], model_names[1]
            )
            
            # 비교 시각화
            self.plot_model_comparison(model_comparison)
        
        # 보고서 생성
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            created_at=datetime.now().isoformat(),
            dataset_info=dataset_info or {
                'test_samples': len(y_test),
                'n_features': X_test.shape[1],
                'n_classes': len(np.unique(y_test))
            },
            model_metrics=model_metrics,
            model_comparison=model_comparison,
            config_info={
                'class_names': self.config.class_names,
                'sample_rate': self.config.sample_rate,
                'n_mfcc': self.config.n_mfcc
            }
        )
        
        # 보고서 저장
        if save_report:
            report_path = self.save_evaluation_report(report)
            self.logger.info(f"평가 보고서 저장됨: {report_path}")
        
        self.logger.info("=== 종합 평가 보고서 생성 완료 ===")
        return report
    
    def save_evaluation_report(self, report: EvaluationReport) -> str:
        """
        평가 보고서를 JSON 파일로 저장합니다.
        
        Parameters:
        -----------
        report : EvaluationReport
            평가 보고서
            
        Returns:
        --------
        str
            저장된 파일 경로
        """
        report_dir = os.path.join(self.config.model_output_dir, "evaluation_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"{report.evaluation_id}.json")
        
        # numpy 배열을 리스트로 변환하여 JSON 직렬화 가능하도록 함
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # 보고서를 딕셔너리로 변환하고 numpy 배열 처리
        report_dict = asdict(report)
        
        # 혼동 행렬 변환
        for model_name, metrics in report_dict['model_metrics'].items():
            metrics['confusion_matrix'] = convert_numpy(metrics['confusion_matrix'])
        
        # JSON 저장
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        return report_path
    
    def load_evaluation_report(self, report_path: str) -> EvaluationReport:
        """
        저장된 평가 보고서를 로드합니다.
        
        Parameters:
        -----------
        report_path : str
            보고서 파일 경로
            
        Returns:
        --------
        EvaluationReport
            로드된 평가 보고서
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            report_dict = json.load(f)
        
        # numpy 배열 복원
        for model_name, metrics in report_dict['model_metrics'].items():
            metrics['confusion_matrix'] = np.array(metrics['confusion_matrix'])
        
        # TODO: 완전한 객체 복원을 위해서는 추가 처리 필요
        self.logger.info(f"평가 보고서 로드됨: {report_path}")
        return report_dict
    
    def get_evaluation_summary(self, report: EvaluationReport) -> Dict[str, Any]:
        """
        평가 보고서의 요약 정보를 반환합니다.
        
        Parameters:
        -----------
        report : EvaluationReport
            평가 보고서
            
        Returns:
        --------
        Dict[str, Any]
            요약 정보
        """
        summary = {
            'evaluation_id': report.evaluation_id,
            'total_models': len(report.model_metrics),
            'dataset_info': report.dataset_info,
            'model_performance': {}
        }
        
        # 모델별 주요 성능 지표
        for model_name, metrics in report.model_metrics.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics.accuracy,
                'f1_macro': metrics.f1_macro,
                'precision_macro': metrics.precision_macro,
                'recall_macro': metrics.recall_macro
            }
        
        # 최고 성능 모델
        if report.model_metrics:
            best_model = max(report.model_metrics.items(), 
                           key=lambda x: x[1].f1_macro)
            summary['best_model'] = {
                'name': best_model[0],
                'f1_macro': best_model[1].f1_macro,
                'accuracy': best_model[1].accuracy
            }
        
        # 모델 비교 결과
        if report.model_comparison:
            summary['model_comparison'] = {
                'better_model': report.model_comparison.better_model,
                'accuracy_diff': report.model_comparison.accuracy_diff,
                'f1_macro_diff': report.model_comparison.f1_macro_diff,
                'statistically_significant': (
                    report.model_comparison.accuracy_ttest['pvalue'] < 0.05 or
                    report.model_comparison.f1_ttest['pvalue'] < 0.05
                )
            }
        
        return summary