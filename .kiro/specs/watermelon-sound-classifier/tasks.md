# 구현 계획

- [x] 1. 프로젝트 구조 및 핵심 구성 설정 ✅ **완료 및 업데이트됨**
  - [x] 표준화된 데이터 디렉토리 구조 생성 (pre-split: data/raw/train|validation|test/, retail noise: data/noise/environmental/retail/homeplus|emart/)
  - [x] 모듈, 소스코드, 구성파일을 위한 디렉토리 구조 생성 (src/, config/, logs/, results/)
  - [x] 데이터클래스를 사용한 구성 관리 시스템 구현 (Config 클래스: pre-split 지원, 재귀적 소음 파일 검색, 유연한 증강 설정)
  - [x] 디버깅 및 모니터링을 위한 로깅 인프라 설정 (src/utils/logger.py, LoggerMixin)
  - [x] 모든 필요한 의존성을 포함한 requirements.txt 생성 (librosa, scikit-learn, coremltools 등)
  - [x] 데이터 파일 배치 가이드 및 README 문서 생성 (DATA_PLACEMENT_GUIDE.md, README.md: retail 환경 특화)
  - [x] 구성 검증 및 테스트 (test_config.py: 24개 소음 파일 검증 완료)
  - _요구사항: 8.1, 8.4, 8.5_ ✅

- [x] 2. 오디오 특징 추출 모듈 구현 ✅ **완료**
  - [x] 2.1 librosa를 사용한 핵심 특징 추출 함수 생성
    - [x] 오디오를 로드하고 모든 필요한 특징을 추출하는 extract_features() 함수 작성 (src/audio/feature_extraction.py)
    - [x] 13개 계수를 가진 MFCC 추출 구현 (AudioFeatureExtractor.extract_mfcc)
    - [x] Mel Spectrogram 통계 계산 추가 (평균, 표준편차) (AudioFeatureExtractor.extract_mel_spectrogram_stats)
    - [x] Spectral Centroid, Spectral Rolloff, Zero Crossing Rate 추출 포함 (AudioFeatureExtractor.extract_spectral_features)
    - [x] 12차원 Chroma Features 추출 구현 (AudioFeatureExtractor.extract_chroma_features)
    - [x] 특징 벡터 연결 생성 및 numpy 배열로 반환 (FeatureVector.to_array, 30차원 벡터)
    - _요구사항: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_ ✅

  - [x] 2.2 특징 추출 검증 및 오류 처리 추가
    - [x] 오디오 파일 검증 구현 (형식, 손상, 무음 감지) (AudioFeatureExtractor.validate_audio_file)
    - [x] librosa 로딩 실패에 대한 오류 처리 추가 (AudioFeatureExtractor.load_audio)
    - [x] 특징 벡터 형태 및 값 범위 검증 생성 (extract_features: NaN/infinity 처리, 크기 검증)
    - [x] 샘플 오디오 파일을 사용한 특징 추출 단위 테스트 작성 (test_feature_extraction.py)
    - _요구사항: 1.1, 1.7, 7.5_ ✅

- [x] 3. 데이터 증강 모듈 구현 ✅ **완료**
  - [x] 3.1 SNR 제어를 통한 소음 증강 함수 생성
    - [x] 깨끗한 오디오를 소음 파일과 혼합하는 augment_noise() 함수 작성 (src/data/augmentation.py:492)
    - [x] SNR 계산 및 소음 스케일링 알고리즘 구현 (AudioAugmentor.calculate_snr, scale_noise_for_snr)
    - [x] 다양한 소음 유형과 SNR 레벨 지원 추가 (config.snr_levels, 홈플러스/이마트 등)
    - [x] 증강된 샘플을 위한 파일 명명 규칙 생성 (원본명_noise_소음명_snr값dB.wav)
    - _요구사항: 2.1, 2.2, 2.3, 2.4_ ✅

  - [x] 3.2 증강 배치 처리 및 검증 추가
    - [x] 여러 원본 파일에 대한 배치 처리 구현 (BatchAugmentor.augment_class_directory)
    - [x] 소음 파일이 없는 경우 원본 데이터만 사용하는 대체 로직 추가 (fallback logic)
    - [x] 사용 가능한 소음 파일 수에 따른 동적 증강 배수 조정 구현 (calculate_dynamic_augmentation)
    - [x] 증강된 샘플에 대한 오디오 품질 검증 추가 (validate_augmented_audio: RMS, 클리핑, NaN 검사)
    - [x] 임시 증강 파일에 대한 정리 기능 생성 (cleanup_augmented_files)
    - [x] SNR 계산 및 오디오 혼합에 대한 단위 테스트 작성 (test_augmentation.py)
    - _요구사항: 2.1, 2.2, 2.5, 7.5_ ✅

- [x] 4. 데이터 파이프라인 관리 구현 ✅ **완료**
  - [x] 4.1 데이터 로딩 및 분할 기능 생성
    - [x] data/raw/train/ 디렉토리에서 클래스별 오디오 파일을 로드하는 load_train_data() 함수 작성 (DataPipeline.load_train_data)
    - [x] data/raw/validation/ 디렉토리에서 클래스별 오디오 파일을 로드하는 load_validation_data() 함수 작성 (DataPipeline.load_validation_data)
    - [x] data/raw/test/ 디렉토리에서 클래스별 오디오 파일을 로드하는 load_test_data() 함수 작성 (DataPipeline.load_test_data)
    - [x] data/noise/ 디렉토리에서 사용 가능한 소음 파일을 재귀적으로 로드하는 load_noise_files() 함수 구현 (DataPipeline.load_noise_files)
    - [x] 파일 경로 관리 및 메타데이터 추적 추가 (AudioFile: 크기, 지속시간, SNR 정보 등)
    - [x] 메타데이터 관리를 위한 AudioFile 데이터클래스 생성 (src/data/pipeline.py:18-42)
    - _요구사항: 4.1, 4.5, 8.1_ ✅

  - [x] 4.2 누출 방지를 통한 파이프라인 오케스트레이션 구현
    - [x] 완전한 데이터 흐름을 관리하는 DataPipeline 클래스 생성 (src/data/pipeline.py:61-608)
    - [x] 적절한 순서 구현: 분할 → 소음 파일 탐색 → 훈련만 증강 → 특징 추출 (run_complete_pipeline)
    - [x] 소음 파일이 없는 폴더를 자동으로 건너뛰는 로직 구현 (augment_training_data: fallback 로직)
    - [x] 증강 중 테스트/검증 세트가 건드려지지 않도록 보장하는 검증 추가 (validate_data_integrity)
    - [x] 데이터 누출이 발생하지 않음을 확인하는 통합 테스트 작성 (test_pipeline.py: test_data_leakage_prevention)
    - _요구사항: 4.1, 4.2, 4.3, 4.4, 4.5_ ✅

- [x] 5. 머신러닝 훈련 모듈 구현 ✅ **완료**
  - [x] 5.1 모델 구성 및 초기화 생성
    - [x] 하이퍼파라미터 그리드를 가진 SVM 및 Random Forest 모델 구성 정의 (ModelConfig 클래스, design.md 명세)
    - [x] 모델 초기화를 가진 ModelTrainer 클래스 구현 (src/ml/training.py:70-557)
    - [x] 두 모델에 대한 하이퍼파라미터 그리드 정의 추가 (_initialize_model_configs: SVM C/gamma, RF n_estimators/max_depth/min_samples_split)
    - [x] 훈련된 모델 저장/로딩을 위한 모델 지속성 기능 생성 (save_models, load_model, ModelArtifact 클래스)
    - _요구사항: 3.1, 3.2, 3.4, 7.1_ ✅

  - [x] 5.2 교차 검증 훈련 파이프라인 구현
    - [x] 5-fold 교차 검증과 함께 GridSearchCV를 사용하는 train_with_cv() 메서드 작성 (src/ml/training.py:262-307)
    - [x] 클래스 분포를 유지하기 위한 계층화된 K-fold 구현 (StratifiedKFold with shuffle=True)
    - [x] SVM과 Random Forest 모두에 대한 하이퍼파라미터 최적화 추가 (GridSearchCV with param_grid)
    - [x] 교차 검증 점수 추적 및 최적 매개변수 선택 생성 (TrainingResult 클래스, best_params/best_score/cv_scores)
    - [x] 포괄적인 테스트 스크립트 작성 (test_training.py: 7개 테스트 함수)
    - _요구사항: 5.1, 5.2, 5.3, 5.4_ ✅

- [x] 6. 모델 평가 및 메트릭 구현 ✅ **완료**
  - [x] 6.1 포괄적인 평가 기능 생성
    - [x] 테스트 세트 평가를 위한 evaluate_model() 메서드 작성 (src/ml/evaluation.py:110-240)
    - [x] 정확도, 정밀도, 재현율, F1-score 계산 구현 (ClassificationMetrics 클래스, macro/weighted 평균)
    - [x] 혼동 행렬 생성 및 시각화 추가 (plot_confusion_matrix, seaborn heatmap)
    - [x] 클래스별 및 매크로 평균 메트릭 보고 생성 (class_precision, class_recall, class_f1)
    - [x] ROC AUC 및 PR AUC 계산 추가 (다중 클래스 지원)
    - _요구사항: 6.1, 6.2, 6.3_ ✅

  - [x] 6.2 성능 비교 및 보고 추가
    - [x] SVM과 Random Forest 간의 모델 비교 기능 구현 (compare_models 메서드)
    - [x] 통계적 유의성 테스트를 포함한 상세한 성능 보고서 생성 (t-test, ModelComparison 클래스)
    - [x] 혼동 행렬 및 성능 메트릭에 대한 시각화 추가 (plot_model_comparison, 4개 subplot)
    - [x] 종합 평가 보고서 생성 기능 구현 (EvaluationReport, JSON 저장/로딩)
    - [x] 알려진 데이터셋을 사용한 포괄적인 평가 테스트 작성 (test_evaluation.py: 8개 테스트 함수)
    - _요구사항: 3.3, 6.1, 6.2, 6.3_ ✅

- [x] 7. 모델 저장 및 형식 변환 구현 ✅ **완료**
  - [x] 7.1 모델 저장 기능 생성
    - [x] 훈련된 모델을 pickle(.pkl) 형식으로 저장하는 save_pickle_model() 메서드 구현 (src/ml/model_converter.py:53-120)
    - [x] 모델과 함께 전처리 파라미터 및 특징 추출 설정 저장 (메타데이터 JSON 파일, 특징 추출 구성 포함)
    - [x] ConversionResult 및 CoreMLModelInfo 데이터클래스를 사용한 모델 메타데이터 관리
    - [x] 모델 버전 관리 및 타임스탬프 추가 기능 구현 (created_at, model_version 필드)
    - [x] ModelTrainer 클래스에 이미 구현된 save_models() 메서드와 통합
    - _요구사항: 7.1, 7.2_ ✅

  - [x] 7.2 Core ML 변환 기능 구현
    - [x] coremltools를 사용한 pickle 모델의 Core ML(.mlmodel) 변환 기능 구현 (convert_to_coreml 메서드)
    - [x] 입력 특징 형태와 출력 클래스 정보의 올바른 매핑 보장 (30차원 입력, 3클래스 출력)
    - [x] 변환된 모델의 예측 결과 일치성 검증 기능 추가 (validate_model_conversion, 10개 샘플 테스트)
    - [x] Core ML 모델 사용 예제 및 로딩 방법 문서화 (docs/COREML_USAGE.md: Python, iOS/Swift 예제)
    - [x] 포괄적인 테스트 스크립트 작성 (test_model_converter.py: 7개 테스트 함수)
    - [x] 변환 요약 및 메타데이터 생성 기능 추가 (get_conversion_summary, create_model_metadata)
    - _요구사항: 7.3, 7.4, 7.5_ ✅

- [x] 8. 메인 실행 파이프라인 생성 ✅ **완료**
  - [x] 8.1 엔드투엔드 파이프라인 오케스트레이션 구현 ✅ **완료**
    - [x] 모든 구성요소를 조정하는 메인 실행 스크립트 작성 (main.py: WatermelonClassificationPipeline 클래스, 4단계 파이프라인)
    - [x] 파이프라인 구성을 위한 명령줄 인터페이스 구현 (argparse: --skip-augmentation, --cv-folds, --no-coreml, --resume, --status, --dry-run)
    - [x] 파이프라인 전반에 걸친 진행 추적 및 로깅 추가 (단계별 시간 측정, 실행 통계, 상세 로깅)
    - [x] 중단된 실행을 재개하기 위한 체크포인트 기능 생성 (PipelineCheckpoint 클래스, JSON 기반 상태 저장/복원)
    - _요구사항: 8.2, 8.3, 8.4_ ✅

  - [x] 8.2 파이프라인 검증 및 테스트 추가 ✅ **완료**
    - [x] 완전한 파이프라인에 대한 통합 테스트 작성 (test_integration.py: 7개 테스트 함수, 가짜 데이터 생성, 엔드투엔드 테스트)
    - [x] 각 파이프라인 단계에서 데이터 무결성 검사 구현 (src/utils/data_integrity.py: DataIntegrityChecker 클래스, 8가지 검사 항목)
    - [x] 성능 벤치마킹 및 메모리 사용량 모니터링 추가 (src/utils/performance_monitor.py: PerformanceMonitor 클래스, 메모리/CPU 측정)
    - [x] 메인 파이프라인에 품질 관리 기능 통합 (main.py: --no-integrity-checks, --no-performance-monitoring 옵션, 자동 보고서 생성)
    - _요구사항: 4.5, 8.5_ ✅

- [ ] 9. 유틸리티 함수 및 헬퍼 구현
  - [x] 9.1 공통 유틸리티 함수 생성 ✅ **완료**
    - [x] 오디오 및 메타데이터 처리를 위한 파일 I/O 유틸리티 작성 (src/utils/file_utils.py: FileUtils, JsonUtils, PickleUtils, AudioFileUtils 클래스)
    - [x] 특징 처리를 위한 배열 조작 헬퍼 구현 (ArrayUtils 클래스: 정규화, 분할, 배치 처리, 이상치 제거, 특징 중요도)
    - [x] 데이터 탐색 및 결과를 위한 시각화 유틸리티 추가 (VisualizationUtils 클래스: 특징 분포, 상관관계, 클래스 분포, 특징 중요도 시각화)
    - [x] 대용량 데이터셋 처리를 위한 메모리 관리 유틸리티 생성 (MemoryUtils 클래스: 메모리 모니터링, 청크 처리, 효율적 연산, 가비지 컬렉션)
    - _요구사항: 8.1, 8.4, 8.5_ ✅

  - [x] 9.2 문서화 및 예제 추가 ✅ **완료** (2024년 완료)
    - [x] 모든 함수와 클래스에 대한 포괄적인 독스트링 작성 (src/ 모든 모듈 독스트링 추가 완료)
    - [x] 사용 예제 및 샘플 코드 스니펫 생성 (docs/USAGE_EXAMPLES.md: 9개 카테고리, 40+ 예제)
    - [x] 설치 및 사용 지침이 포함된 README 업데이트 (빠른 시작, 고급 사용법, 문제 해결 포함)
    - [x] 모든 공개 인터페이스에 대한 API 문서 작성 (docs/API_REFERENCE.md: 완전한 API 레퍼런스)
    - [x] pickle 및 Core ML 모델 사용 예제 추가 (docs/MODEL_USAGE_EXAMPLES.md: Python/Swift/Objective-C 예제)
    - _요구사항: 8.2, 8.3, 8.5, 7.5_ ✅

- [x] 10. 포괄적인 테스트 스위트 생성 ✅ **완료**
  - [x] 10.1 모든 구성요소에 대한 단위 테스트 작성 ✅ **완료**
    - [x] 다양한 오디오 형식을 사용한 특징 추출 단위 테스트 생성 (tests/unit/audio/test_feature_extraction_extended.py: 극단적 경우, 다양한 형식, 에러 처리)
    - [x] 다양한 SNR 레벨을 사용한 데이터 증강 테스트 작성 (tests/unit/data/test_augmentation_extended.py: 극단적 SNR, 다양한 노이즈 유형, 성능 테스트)
    - [x] 데이터 파이프라인 무결성 및 누출 방지 테스트 추가 (기존 test_pipeline.py에 포함, 데이터 무결성 검증)
    - [x] 모델 훈련 및 평가 함수에 대한 테스트 구현 (기존 test_training.py, test_evaluation.py에 포함)
    - [x] 모델 저장 및 Core ML 변환 기능에 대한 테스트 추가 (기존 test_model_converter.py에 포함)
    - [x] 유틸리티 모듈 및 설정 테스트 추가 (tests/unit/utils/test_file_utils.py, tests/unit/test_config.py)
    - _요구사항: 8.5, 7.1, 7.3_ ✅

  - [x] 10.2 통합 및 성능 테스트 추가 ✅ **완료**
    - [x] 샘플 데이터셋을 사용한 엔드투엔드 통합 테스트 작성 (기존 tests/integration/test_integration.py에 포함, 전체 파이프라인 검증)
    - [x] 처리 속도 및 메모리 사용량에 대한 성능 벤치마크 생성 (tests/performance/test_benchmarks.py: 종합 성능 벤치마크)
    - [x] 손상되거나 비정상적인 오디오 파일에 대한 극단적 경우 테스트 추가 (feature_extraction_extended.py에 포함)
    - [x] 실행 간 일관된 결과를 보장하는 회귀 테스는 재현성 테스트 구현 (각 테스트 모듈에 재현성 테스트 포함)
    - [x] pickle과 Core ML 모델 간 예측 결과 일치성 테스트 추가 (test_model_converter.py에 포함)
    - [x] 테스트 구조화 및 실행 자동화 (tests/ 디렉토리 구조화, pytest.ini, run_tests.py, conftest.py)
    - _요구사항: 8.5, 7.4_ ✅

- [x] 11. 구현 최적화 및 완료 ✅ **완료**
  - [x] 11.1 성능 최적화 및 확장성 ✅ **완료**
    - [x] 배치 처리를 위한 특징 추출 최적화 (src/audio/batch_processor.py: OptimizedFeatureExtractor, BatchFeatureProcessor, StreamingFeatureProcessor)
    - [x] 데이터 증강을 위한 병렬 처리 구현 (src/data/parallel_augmentor.py: ParallelBatchAugmentor, OptimizedAudioAugmentor, StreamingAugmentor)
    - [x] 대용량 데이터셋을 위한 메모리 효율적 처리 추가 (src/data/large_dataset_processor.py: StreamingDatasetProcessor, ChunkedFileProcessor, MemoryMonitor)
    - [x] 다양한 하드웨어 설정을 위한 구성 옵션 생성 (src/config/hardware_config.py: HardwareDetector, PerformancePresetManager, HardwareConfigManager)
    - [x] 통합 최적화 시스템 구현 (src/optimization/integrated_optimizer.py: IntegratedOptimizer, 자동 하드웨어 감지 및 최적화)
    - _요구사항: 3.4, 8.4_ ✅

  - [x] 11.2 최종 검증 및 정리 ✅ **완료**
    - [x] 여러 데이터셋을 사용한 완전한 파이프라인 검증 실행 (validation/comprehensive_pipeline_validation.py: 포괄적 파이프라인 검증기, 8개 검증 카테고리)
    - [x] 포괄적인 테스트를 통해 모든 요구사항이 충족되었는지 확인 (validation/requirements_verification.py: Python 패키지, 시스템, 파일 구조, 설정 검증)
    - [x] 임시 파일 정리 및 리소스 사용량 최적화 (src/utils/resource_cleanup.py: 자동 리소스 정리, 메모리 최적화, 가비지 컬렉션)
    - [x] 최종 문서화 및 사용 예제 생성 (docs/FINAL_DEPLOYMENT_GUIDE.md: 종합 배포 가이드, 플랫폼별 배포 방법, 최적화 전략)
    - [x] pickle 및 Core ML 모델의 배포 준비 상태 확인 (validation/deployment_readiness_checker.py: 모델 무결성, 성능, 호환성, 보안 검증)
    - _요구사항: 6.3, 8.2, 8.3, 8.5, 7.5_ ✅