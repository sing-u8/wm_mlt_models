"""
머신러닝 훈련 모듈

이 모듈은 scikit-learn을 사용하여 SVM과 Random Forest 분류 모델을 
훈련하고 평가하는 기능을 제공합니다.
"""

import os
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from ..utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class ModelConfig:
    """
    모델 구성을 담는 데이터 클래스.
    """
    model_type: str
    model_params: Dict[str, Any]
    param_grid: Dict[str, List[Any]]
    cv_folds: int = 5
    scoring: str = 'f1_macro'
    random_state: int = 42


@dataclass
class TrainingResult:
    """
    훈련 결과를 담는 데이터 클래스.
    """
    model_type: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    training_time: float
    n_features: int
    n_samples: int
    feature_importance: Optional[List[float]] = None  # Random Forest만


@dataclass
class ModelArtifact:
    """
    모델 아티팩트 메타데이터를 담는 데이터 클래스.
    """
    model_type: str
    model_path: str
    config_path: str
    created_at: str
    training_result: TrainingResult
    feature_extraction_config: Dict[str, Any]
    class_names: List[str]
    feature_names: List[str]
    model_version: str = "1.0"


class ModelTrainer(LoggerMixin):
    """
    머신러닝 모델 훈련을 위한 클래스.
    
    design.md에 명시된 인터페이스를 구현합니다.
    """
    
    def __init__(self, config=None):
        """
        모델 구성으로 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        
        # 모델 구성 정의
        self.models = self._initialize_model_configs()
        
        # 훈련된 모델 저장
        self.trained_models = {}
        self.training_results = {}
        self.model_artifacts = {}
        
        self.logger.info(f"ModelTrainer 초기화됨")
        self.logger.info(f"사용 가능한 모델: {list(self.models.keys())}")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """
        design.md 명세에 따른 모델 구성을 초기화합니다.
        
        Returns:
        --------
        Dict[str, ModelConfig]
            모델별 구성 딕셔너리
        """
        model_configs = {}
        
        # SVM 구성 (design.md 명세)
        svm_config = ModelConfig(
            model_type="svm",
            model_params={
                "kernel": "rbf",
                "random_state": 42,
                "probability": True  # predict_proba를 위해 필요
            },
            param_grid={
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01]
            }
        )
        model_configs["svm"] = svm_config
        
        # Random Forest 구성 (design.md 명세)
        rf_config = ModelConfig(
            model_type="random_forest",
            model_params={
                "random_state": 42,
                "n_jobs": -1  # 병렬 처리
            },
            param_grid={
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        )
        model_configs["random_forest"] = rf_config
        
        return model_configs
    
    def _create_model_instance(self, model_type: str, params: Dict[str, Any]):
        """
        모델 인스턴스를 생성합니다.
        
        Parameters:
        -----------
        model_type : str
            모델 타입 ("svm" 또는 "random_forest")
        params : Dict[str, Any]
            모델 파라미터
            
        Returns:
        --------
        sklearn model instance
            생성된 모델 인스턴스
        """
        if model_type == "svm":
            return SVC(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    def train_single_model(self, model_type: str, X_train: np.ndarray, 
                          y_train: np.ndarray, cv_folds: int = 5) -> TrainingResult:
        """
        단일 모델을 교차 검증으로 훈련합니다.
        
        Parameters:
        -----------
        model_type : str
            훈련할 모델 타입
        X_train : np.ndarray
            훈련 특징 데이터
        y_train : np.ndarray
            훈련 라벨 데이터
        cv_folds : int
            교차 검증 폴드 수
            
        Returns:
        --------
        TrainingResult
            훈련 결과
        """
        if model_type not in self.models:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
        
        model_config = self.models[model_type]
        
        self.logger.info(f"=== {model_type.upper()} 모델 훈련 시작 ===")
        self.logger.info(f"데이터 형태: {X_train.shape}")
        self.logger.info(f"클래스 분포: {np.bincount(y_train)}")
        self.logger.info(f"하이퍼파라미터 그리드: {model_config.param_grid}")
        
        start_time = datetime.now()
        
        # 기본 모델 인스턴스 생성
        base_model = self._create_model_instance(model_type, model_config.model_params)
        
        # 계층화된 K-Fold 교차 검증 설정
        stratified_kfold = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=42
        )
        
        # GridSearchCV 설정
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=model_config.param_grid,
            scoring=model_config.scoring,
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        try:
            # 그리드 서치 실행
            self.logger.info("GridSearchCV 실행 중...")
            grid_search.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 결과 수집
            best_model = grid_search.best_estimator_
            cv_scores = grid_search.cv_results_['mean_test_score']
            
            # 특징 중요도 추출 (Random Forest만)
            feature_importance = None
            if model_type == "random_forest" and hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_.tolist()
            
            # 훈련 결과 생성
            training_result = TrainingResult(
                model_type=model_type,
                best_params=grid_search.best_params_,
                best_score=grid_search.best_score_,
                cv_scores=cv_scores.tolist(),
                training_time=training_time,
                n_features=X_train.shape[1],
                n_samples=X_train.shape[0],
                feature_importance=feature_importance
            )
            
            # 모델 저장
            self.trained_models[model_type] = best_model
            self.training_results[model_type] = training_result
            
            self.logger.info(f"=== {model_type.upper()} 모델 훈련 완료 ===")
            self.logger.info(f"최적 파라미터: {grid_search.best_params_}")
            self.logger.info(f"최적 CV 점수: {grid_search.best_score_:.4f}")
            self.logger.info(f"훈련 시간: {training_time:.2f}초")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"{model_type} 모델 훈련 실패: {e}")
            raise
    
    def train_with_cv(self, X_train: np.ndarray, y_train: np.ndarray, 
                      cv_folds: int = 5) -> Dict[str, TrainingResult]:
        """
        교차 검증으로 모든 모델을 훈련합니다.
        
        design.md에 명시된 인터페이스를 구현합니다.
        
        Parameters:
        -----------
        X_train : np.ndarray
            훈련 특징 데이터
        y_train : np.ndarray  
            훈련 라벨 데이터
        cv_folds : int
            교차 검증 폴드 수
            
        Returns:
        --------
        Dict[str, TrainingResult]
            모델별 훈련 결과
        """
        self.logger.info("🤖 모든 모델 교차 검증 훈련 시작 🤖")
        self.logger.info(f"총 데이터: {X_train.shape[0]}개 샘플, {X_train.shape[1]}개 특징")
        self.logger.info(f"교차 검증: {cv_folds}-fold")
        
        all_results = {}
        
        # 각 모델별로 훈련 수행
        for model_type in self.models.keys():
            try:
                result = self.train_single_model(model_type, X_train, y_train, cv_folds)
                all_results[model_type] = result
                
            except Exception as e:
                self.logger.error(f"모델 {model_type} 훈련 실패: {e}")
                continue
        
        # 결과 요약
        self.logger.info("🤖 모든 모델 훈련 완료 🤖")
        self.logger.info("모델별 최고 CV 점수:")
        
        for model_type, result in all_results.items():
            self.logger.info(f"  {model_type}: {result.best_score:.4f} "
                           f"(훈련 시간: {result.training_time:.1f}초)")
        
        return all_results
    
    def evaluate_single_model(self, model_type: str, X_test: np.ndarray, 
                             y_test: np.ndarray) -> Dict[str, float]:
        """
        단일 모델을 테스트 세트에서 평가합니다.
        
        Parameters:
        -----------
        model_type : str
            평가할 모델 타입
        X_test : np.ndarray
            테스트 특징 데이터
        y_test : np.ndarray
            테스트 라벨 데이터
            
        Returns:
        --------
        Dict[str, float]
            평가 메트릭들
        """
        if model_type not in self.trained_models:
            raise ValueError(f"모델 {model_type}이 훈련되지 않았습니다.")
        
        model = self.trained_models[model_type]
        
        # 예측 수행
        y_pred = model.predict(X_test)
        
        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # 클래스별 메트릭
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.config.class_names,
                                           output_dict=True, zero_division=0)
        
        # 클래스별 메트릭을 전체 메트릭에 추가
        for class_name in self.config.class_names:
            if class_name in class_report:
                metrics[f'{class_name}_precision'] = class_report[class_name]['precision']
                metrics[f'{class_name}_recall'] = class_report[class_name]['recall']
                metrics[f'{class_name}_f1'] = class_report[class_name]['f1-score']
        
        self.logger.info(f"{model_type} 평가 완료:")
        self.logger.info(f"  정확도: {metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        self.logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def evaluate_final(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        테스트 세트에서 훈련된 모든 모델을 평가합니다.
        
        design.md에 명시된 인터페이스를 구현합니다.
        
        Parameters:
        -----------
        X_test : np.ndarray
            테스트 특징 데이터
        y_test : np.ndarray
            테스트 라벨 데이터
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            모델별 평가 메트릭들
        """
        self.logger.info("📊 최종 모델 평가 시작 📊")
        self.logger.info(f"테스트 데이터: {X_test.shape[0]}개 샘플, {X_test.shape[1]}개 특징")
        
        all_metrics = {}
        
        for model_type in self.trained_models.keys():
            try:
                metrics = self.evaluate_single_model(model_type, X_test, y_test)
                all_metrics[model_type] = metrics
                
            except Exception as e:
                self.logger.error(f"모델 {model_type} 평가 실패: {e}")
                continue
        
        # 모델 비교 (requirements.md 요구사항 3.5)
        if len(all_metrics) >= 2:
            self.logger.info("📊 모델 성능 비교 📊")
            
            comparison_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
            for metric in comparison_metrics:
                self.logger.info(f"{metric.upper()}:")
                for model_type, metrics in all_metrics.items():
                    self.logger.info(f"  {model_type}: {metrics[metric]:.4f}")
        
        self.logger.info("📊 최종 모델 평가 완료 📊")
        
        return all_metrics
    
    def save_models(self, output_dir: str = None) -> Dict[str, str]:
        """
        훈련된 모델을 pickle 형식으로 저장합니다.
        
        design.md에 명시된 인터페이스를 구현합니다.
        
        Parameters:
        -----------
        output_dir : str, optional
            출력 디렉토리. None이면 기본 모델 디렉토리 사용
            
        Returns:
        --------
        Dict[str, str]
            모델별 저장된 파일 경로
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.model_output_dir, "pickle")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"💾 모델 저장 시작: {output_dir}")
        
        saved_paths = {}
        
        for model_type, model in self.trained_models.items():
            try:
                # 모델 파일 경로
                model_filename = f"{model_type}_model.pkl"
                model_path = os.path.join(output_dir, model_filename)
                
                # 모델 저장
                joblib.dump(model, model_path)
                saved_paths[model_type] = model_path
                
                # 모델 아티팩트 생성 (requirements.md 요구사항 7.2)
                artifact = ModelArtifact(
                    model_type=model_type,
                    model_path=model_path,
                    config_path=os.path.join(output_dir, f"{model_type}_config.json"),
                    created_at=datetime.now().isoformat(),
                    training_result=self.training_results[model_type],
                    feature_extraction_config={
                        'sample_rate': self.config.sample_rate,
                        'hop_length': self.config.hop_length,
                        'n_mfcc': self.config.n_mfcc,
                        'n_chroma': self.config.n_chroma
                    },
                    class_names=self.config.class_names,
                    feature_names=[f'feature_{i}' for i in range(30)]  # 30차원 특징
                )
                
                # 아티팩트 메타데이터 저장
                with open(artifact.config_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(artifact), f, indent=2, ensure_ascii=False)
                
                self.model_artifacts[model_type] = artifact
                
                self.logger.info(f"  {model_type}: 저장 완료 ({os.path.basename(model_path)})")
                
            except Exception as e:
                self.logger.error(f"모델 {model_type} 저장 실패: {e}")
                continue
        
        self.logger.info(f"💾 모델 저장 완료: {len(saved_paths)}개 모델")
        
        return saved_paths
    
    def load_model(self, model_path: str) -> Any:
        """
        저장된 모델을 로드합니다.
        
        Parameters:
        -----------
        model_path : str
            모델 파일 경로
            
        Returns:
        --------
        Any
            로드된 모델 객체
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"모델 로드 완료: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패 {model_path}: {e}")
            raise
    
    def get_feature_importance(self, model_type: str) -> Optional[List[Tuple[str, float]]]:
        """
        모델의 특징 중요도를 반환합니다 (Random Forest만 지원).
        
        Parameters:
        -----------
        model_type : str
            모델 타입
            
        Returns:
        --------
        Optional[List[Tuple[str, float]]]
            (특징명, 중요도) 튜플 리스트
        """
        if model_type not in self.training_results:
            return None
        
        result = self.training_results[model_type]
        if result.feature_importance is None:
            return None
        
        # 특징명과 중요도 매핑
        feature_names = [f'feature_{i}' for i in range(len(result.feature_importance))]
        importance_pairs = list(zip(feature_names, result.feature_importance))
        
        # 중요도 순으로 정렬
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return importance_pairs
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        훈련 요약 정보를 반환합니다.
        
        Returns:
        --------
        Dict[str, Any]
            훈련 요약 정보
        """
        summary = {
            'total_models': len(self.trained_models),
            'trained_models': list(self.trained_models.keys()),
            'training_results': {}
        }
        
        for model_type, result in self.training_results.items():
            summary['training_results'][model_type] = {
                'best_score': result.best_score,
                'best_params': result.best_params,
                'training_time': result.training_time,
                'n_features': result.n_features,
                'n_samples': result.n_samples
            }
        
        return summary