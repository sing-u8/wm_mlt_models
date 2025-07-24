"""
파일 I/O 및 메타데이터 처리 유틸리티

오디오 파일 처리, 메타데이터 관리, 파일 시스템 작업을 위한 공통 유틸리티 함수들을 제공합니다.
"""

import os
import json
import pickle
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .logger import LoggerMixin


class FileUtils(LoggerMixin):
    """
    파일 I/O 작업을 위한 유틸리티 클래스
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        디렉토리가 존재하지 않으면 생성합니다.
        
        Parameters:
        -----------
        path : Union[str, Path]
            생성할 디렉토리 경로
            
        Returns:
        --------
        Path
            생성된 디렉토리 경로
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        파일의 기본 정보를 수집합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            파일 경로
            
        Returns:
        --------
        Dict[str, Any]
            파일 정보 딕셔너리
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found', 'path': str(file_path)}
        
        stat_info = file_path.stat()
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size_bytes': stat_info.st_size,
            'size_mb': stat_info.st_size / (1024 * 1024),
            'created_at': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'is_file': file_path.is_file(),
            'is_directory': file_path.is_dir()
        }
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        파일의 해시값을 계산합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            파일 경로
        algorithm : str
            해시 알고리즘 ('md5', 'sha1', 'sha256')
            
        Returns:
        --------
        str
            파일 해시값
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def find_files(directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = True,
                   file_types: Optional[List[str]] = None) -> List[Path]:
        """
        지정된 패턴에 맞는 파일들을 찾습니다.
        
        Parameters:
        -----------
        directory : Union[str, Path]
            검색할 디렉토리
        pattern : str
            파일 패턴 (glob 패턴)
        recursive : bool
            하위 디렉토리까지 검색할지 여부
        file_types : Optional[List[str]]
            파일 확장자 필터 (예: ['.wav', '.mp3'])
            
        Returns:
        --------
        List[Path]
            찾은 파일 경로 목록
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # 파일만 필터링
        files = [f for f in files if f.is_file()]
        
        # 파일 타입 필터링
        if file_types:
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in file_types]
            files = [f for f in files if f.suffix.lower() in file_types]
        
        return sorted(files)
    
    @staticmethod
    def copy_file(source: Union[str, Path], 
                  destination: Union[str, Path],
                  create_dirs: bool = True) -> Path:
        """
        파일을 복사합니다.
        
        Parameters:
        -----------
        source : Union[str, Path]
            원본 파일 경로
        destination : Union[str, Path]
            대상 파일 경로
        create_dirs : bool
            대상 디렉토리를 자동 생성할지 여부
            
        Returns:
        --------
        Path
            복사된 파일 경로
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if create_dirs:
            FileUtils.ensure_directory(destination.parent)
        
        shutil.copy2(source, destination)
        return destination
    
    @staticmethod
    def move_file(source: Union[str, Path], 
                  destination: Union[str, Path],
                  create_dirs: bool = True) -> Path:
        """
        파일을 이동합니다.
        
        Parameters:
        -----------
        source : Union[str, Path]
            원본 파일 경로
        destination : Union[str, Path]
            대상 파일 경로
        create_dirs : bool
            대상 디렉토리를 자동 생성할지 여부
            
        Returns:
        --------
        Path
            이동된 파일 경로
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if create_dirs:
            FileUtils.ensure_directory(destination.parent)
        
        shutil.move(str(source), str(destination))
        return destination
    
    @staticmethod
    def delete_file(file_path: Union[str, Path], 
                    safe_delete: bool = True) -> bool:
        """
        파일을 삭제합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            삭제할 파일 경로
        safe_delete : bool
            안전 삭제 모드 (존재하지 않는 파일에 대해 예외 발생하지 않음)
            
        Returns:
        --------
        bool
            삭제 성공 여부
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            if safe_delete:
                return True
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
    
    @staticmethod
    def clean_directory(directory: Union[str, Path], 
                       pattern: str = "*",
                       exclude_patterns: Optional[List[str]] = None,
                       dry_run: bool = False) -> List[Path]:
        """
        디렉토리를 정리합니다.
        
        Parameters:
        -----------
        directory : Union[str, Path]
            정리할 디렉토리
        pattern : str
            삭제할 파일 패턴
        exclude_patterns : Optional[List[str]]
            제외할 패턴 목록
        dry_run : bool
            실제 삭제하지 않고 목록만 반환
            
        Returns:
        --------
        List[Path]
            삭제된 (또는 삭제될) 파일 목록
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        files_to_delete = FileUtils.find_files(directory, pattern, recursive=True)
        
        # 제외 패턴 적용
        if exclude_patterns:
            filtered_files = []
            for file_path in files_to_delete:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_files.append(file_path)
            files_to_delete = filtered_files
        
        if not dry_run:
            deleted_files = []
            for file_path in files_to_delete:
                if FileUtils.delete_file(file_path, safe_delete=True):
                    deleted_files.append(file_path)
            return deleted_files
        
        return files_to_delete


class JsonUtils(LoggerMixin):
    """
    JSON 파일 처리를 위한 유틸리티 클래스
    """
    
    @staticmethod
    def save_json(data: Any, 
                  file_path: Union[str, Path],
                  indent: int = 2,
                  ensure_ascii: bool = False,
                  create_dirs: bool = True) -> Path:
        """
        데이터를 JSON 파일로 저장합니다.
        
        Parameters:
        -----------
        data : Any
            저장할 데이터
        file_path : Union[str, Path]
            저장할 파일 경로
        indent : int
            JSON 들여쓰기
        ensure_ascii : bool
            ASCII 인코딩 강제 여부
        create_dirs : bool
            디렉토리 자동 생성 여부
            
        Returns:
        --------
        Path
            저장된 파일 경로
        """
        file_path = Path(file_path)
        
        if create_dirs:
            FileUtils.ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return file_path
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """
        JSON 파일을 로드합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            로드할 파일 경로
            
        Returns:
        --------
        Any
            로드된 데이터
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def update_json(file_path: Union[str, Path], 
                    updates: Dict[str, Any],
                    create_if_not_exists: bool = True) -> Dict[str, Any]:
        """
        JSON 파일을 업데이트합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            업데이트할 파일 경로
        updates : Dict[str, Any]
            업데이트할 데이터
        create_if_not_exists : bool
            파일이 없으면 생성할지 여부
            
        Returns:
        --------
        Dict[str, Any]
            업데이트된 데이터
        """
        file_path = Path(file_path)
        
        if file_path.exists():
            data = JsonUtils.load_json(file_path)
        elif create_if_not_exists:
            data = {}
        else:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        # 딕셔너리인 경우에만 업데이트
        if isinstance(data, dict):
            data.update(updates)
        else:
            data = updates
        
        JsonUtils.save_json(data, file_path)
        return data


class PickleUtils(LoggerMixin):
    """
    Pickle 파일 처리를 위한 유틸리티 클래스
    """
    
    @staticmethod
    def save_pickle(data: Any, 
                    file_path: Union[str, Path],
                    protocol: int = pickle.HIGHEST_PROTOCOL,
                    create_dirs: bool = True) -> Path:
        """
        데이터를 Pickle 파일로 저장합니다.
        
        Parameters:
        -----------
        data : Any
            저장할 데이터
        file_path : Union[str, Path]
            저장할 파일 경로
        protocol : int
            Pickle 프로토콜 버전
        create_dirs : bool
            디렉토리 자동 생성 여부
            
        Returns:
        --------
        Path
            저장된 파일 경로
        """
        file_path = Path(file_path)
        
        if create_dirs:
            FileUtils.ensure_directory(file_path.parent)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        
        return file_path
    
    @staticmethod
    def load_pickle(file_path: Union[str, Path]) -> Any:
        """
        Pickle 파일을 로드합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            로드할 파일 경로
            
        Returns:
        --------
        Any
            로드된 데이터
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class AudioFileUtils(LoggerMixin):
    """
    오디오 파일 처리를 위한 유틸리티 클래스
    """
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        오디오 파일의 정보를 수집합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            오디오 파일 경로
            
        Returns:
        --------
        Dict[str, Any]
            오디오 파일 정보
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found', 'path': str(file_path)}
        
        info = FileUtils.get_file_info(file_path)
        
        # 오디오 특화 정보 추가
        if LIBROSA_AVAILABLE:
            try:
                # librosa로 오디오 정보 로드 (오디오 데이터는 로드하지 않음)
                duration = librosa.get_duration(path=str(file_path))
                
                # 샘플 레이트 정보 (실제 로드 없이 추정)
                y, sr = librosa.load(str(file_path), sr=None, duration=0.1)  # 0.1초만 로드
                
                info.update({
                    'duration_seconds': duration,
                    'sample_rate': sr,
                    'estimated_samples': int(duration * sr),
                    'audio_format': file_path.suffix.lower(),
                    'librosa_available': True
                })
                
            except Exception as e:
                info.update({
                    'audio_error': str(e),
                    'librosa_available': True
                })
        else:
            info.update({
                'librosa_available': False,
                'audio_format': file_path.suffix.lower()
            })
        
        return info
    
    @staticmethod
    def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        오디오 파일의 유효성을 검사합니다.
        
        Parameters:
        -----------
        file_path : Union[str, Path]
            검사할 오디오 파일 경로
            
        Returns:
        --------
        Dict[str, Any]
            검증 결과
        """
        file_path = Path(file_path)
        
        result = {
            'path': str(file_path),
            'is_valid': False,
            'errors': [],
            'warnings': []
        }
        
        # 파일 존재 확인
        if not file_path.exists():
            result['errors'].append('File does not exist')
            return result
        
        # 파일 크기 확인
        file_size = file_path.stat().st_size
        if file_size == 0:
            result['errors'].append('File is empty')
            return result
        
        if file_size < 1024:  # 1KB 미만
            result['warnings'].append('File size is very small (< 1KB)')
        
        # 파일 확장자 확인
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
        if file_path.suffix.lower() not in audio_extensions:
            result['warnings'].append(f'Unusual audio file extension: {file_path.suffix}')
        
        # librosa로 오디오 파일 검증
        if LIBROSA_AVAILABLE:
            try:
                # 오디오 로드 시도 (짧은 구간만)
                y, sr = librosa.load(str(file_path), duration=1.0)
                
                if len(y) == 0:
                    result['errors'].append('Audio file contains no data')
                    return result
                
                # 기본 통계 확인
                if np.all(y == 0):
                    result['errors'].append('Audio file contains only silence')
                    return result
                
                if np.any(np.isnan(y)):
                    result['errors'].append('Audio file contains NaN values')
                    return result
                
                if np.any(np.isinf(y)):
                    result['errors'].append('Audio file contains infinite values')
                    return result
                
                # RMS 계산
                rms = np.sqrt(np.mean(y**2))
                if rms < 1e-6:
                    result['warnings'].append('Audio signal has very low amplitude')
                
                result.update({
                    'duration': len(y) / sr,
                    'sample_rate': sr,
                    'rms_amplitude': float(rms),
                    'max_amplitude': float(np.max(np.abs(y))),
                    'samples': len(y)
                })
                
                result['is_valid'] = True
                
            except Exception as e:
                result['errors'].append(f'Failed to load audio file: {str(e)}')
        else:
            result['warnings'].append('librosa not available - limited validation')
            result['is_valid'] = True  # 기본 파일 검사만 통과하면 유효로 간주
        
        return result
    
    @staticmethod
    def batch_validate_audio_files(directory: Union[str, Path],
                                  file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        디렉토리 내 모든 오디오 파일을 일괄 검증합니다.
        
        Parameters:
        -----------
        directory : Union[str, Path]
            검증할 디렉토리
        file_types : Optional[List[str]]
            검증할 파일 확장자 목록
            
        Returns:
        --------
        Dict[str, Any]
            일괄 검증 결과
        """
        directory = Path(directory)
        
        if file_types is None:
            file_types = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
        
        audio_files = FileUtils.find_files(directory, recursive=True, file_types=file_types)
        
        results = {
            'directory': str(directory),
            'total_files': len(audio_files),
            'valid_files': 0,
            'invalid_files': 0,
            'files_with_warnings': 0,
            'validation_results': [],
            'summary': {
                'common_errors': {},
                'common_warnings': {},
                'total_duration': 0.0,
                'sample_rates': {}
            }
        }
        
        for file_path in audio_files:
            validation_result = AudioFileUtils.validate_audio_file(file_path)
            results['validation_results'].append(validation_result)
            
            if validation_result['is_valid']:
                results['valid_files'] += 1
                
                # 총 재생 시간 누적
                if 'duration' in validation_result:
                    results['summary']['total_duration'] += validation_result['duration']
                
                # 샘플 레이트 통계
                if 'sample_rate' in validation_result:
                    sr = validation_result['sample_rate']
                    results['summary']['sample_rates'][sr] = results['summary']['sample_rates'].get(sr, 0) + 1
            else:
                results['invalid_files'] += 1
            
            if validation_result['warnings']:
                results['files_with_warnings'] += 1
            
            # 공통 오류/경고 집계
            for error in validation_result['errors']:
                results['summary']['common_errors'][error] = results['summary']['common_errors'].get(error, 0) + 1
            
            for warning in validation_result['warnings']:
                results['summary']['common_warnings'][warning] = results['summary']['common_warnings'].get(warning, 0) + 1
        
        return results


class ArrayUtils(LoggerMixin):
    """
    배열 조작 및 특징 처리 유틸리티 클래스
    """
    
    @staticmethod
    def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        특징을 정규화합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            정규화할 특징 데이터
        method : str
            정규화 방법 ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, Any]]
            (정규화된 데이터, 정규화 파라미터)
        """
        if method == 'standard':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            # 표준편차가 0인 경우 처리
            std = np.where(std == 0, 1, std)
            X_normalized = (X - mean) / std
            params = {'method': 'standard', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            # 범위가 0인 경우 처리
            X_range = X_max - X_min
            X_range = np.where(X_range == 0, 1, X_range)
            X_normalized = (X - X_min) / X_range
            params = {'method': 'minmax', 'min': X_min, 'max': X_max, 'range': X_range}
            
        elif method == 'robust':
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            # MAD가 0인 경우 처리
            mad = np.where(mad == 0, 1, mad)
            X_normalized = (X - median) / mad
            params = {'method': 'robust', 'median': median, 'mad': mad}
            
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")
        
        return X_normalized, params
    
    @staticmethod
    def apply_normalization(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        기존 정규화 파라미터를 적용합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            정규화할 데이터
        params : Dict[str, Any]
            정규화 파라미터
            
        Returns:
        --------
        np.ndarray
            정규화된 데이터
        """
        method = params['method']
        
        if method == 'standard':
            return (X - params['mean']) / params['std']
        elif method == 'minmax':
            return (X - params['min']) / params['range']
        elif method == 'robust':
            return (X - params['median']) / params['mad']
        else:
            raise ValueError(f"지원하지 않는 정규화 방법: {method}")
    
    @staticmethod
    def split_array(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    stratify: bool = True,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        배열을 훈련/테스트 세트로 분할합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            특징 데이터
        y : np.ndarray
            레이블 데이터
        test_size : float
            테스트 세트 비율
        stratify : bool
            계층화 분할 여부
        random_state : int
            랜덤 시드
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            stratify_param = y if stratify else None
            
            return train_test_split(
                X, y, 
                test_size=test_size,
                stratify=stratify_param,
                random_state=random_state
            )
        except ImportError:
            # sklearn이 없는 경우 간단한 분할
            np.random.seed(random_state)
            n_samples = len(X)
            n_test = int(n_samples * test_size)
            
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    @staticmethod
    def batch_process_array(X: np.ndarray, batch_size: int, 
                          process_func: callable, **kwargs) -> np.ndarray:
        """
        배열을 배치 단위로 처리합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            처리할 배열
        batch_size : int
            배치 크기
        process_func : callable
            처리 함수
        **kwargs
            처리 함수에 전달할 추가 인자
            
        Returns:
        --------
        np.ndarray
            처리된 결과
        """
        n_samples = len(X)
        results = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            
            batch_result = process_func(batch_X, **kwargs)
            results.append(batch_result)
        
        return np.concatenate(results, axis=0)
    
    @staticmethod
    def remove_outliers(X: np.ndarray, method: str = 'iqr', 
                       threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상치를 제거합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            데이터 배열
        method : str
            이상치 탐지 방법 ('iqr', 'zscore')
        threshold : float
            임계값
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (정상 데이터, 이상치 마스크)
        """
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            
        elif method == 'zscore':
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outlier_mask = np.any(z_scores > threshold, axis=1)
            
        else:
            raise ValueError(f"지원하지 않는 이상치 탐지 방법: {method}")
        
        clean_data = X[~outlier_mask]
        
        return clean_data, outlier_mask
    
    @staticmethod
    def compute_feature_importance(X: np.ndarray, y: np.ndarray, 
                                 feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        특징 중요도를 계산합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            특징 데이터
        y : np.ndarray
            레이블 데이터
        feature_names : Optional[List[str]]
            특징 이름 목록
            
        Returns:
        --------
        Dict[str, float]
            특징 중요도 딕셔너리
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
            
            importance_scores = mutual_info_classif(X, y, random_state=42)
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            importance_dict = dict(zip(feature_names, importance_scores))
            
            # 중요도 순으로 정렬
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except ImportError:
            # sklearn이 없는 경우 상관계수 사용
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            correlations = []
            for i in range(X.shape[1]):
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0.0)
            
            importance_dict = dict(zip(feature_names, correlations))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class VisualizationUtils(LoggerMixin):
    """
    데이터 시각화 유틸리티 클래스
    """
    
    @staticmethod
    def plot_feature_distribution(X: np.ndarray, feature_names: Optional[List[str]] = None,
                                output_path: Optional[Union[str, Path]] = None,
                                figsize: Tuple[int, int] = (15, 10)):
        """
        특징 분포를 시각화합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            특징 데이터
        feature_names : Optional[List[str]]
            특징 이름 목록
        output_path : Optional[Union[str, Path]]
            저장할 경로
        figsize : Tuple[int, int]
            그래프 크기
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            n_features = X.shape[1]
            if feature_names is None:
                feature_names = [f'Feature {i+1}' for i in range(n_features)]
            
            # 서브플롯 배치 계산
            n_cols = min(4, n_features)
            n_rows = (n_features - 1) // n_cols + 1
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_features):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # 히스토그램 그리기
                ax.hist(X[:, i], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{feature_names[i]}\n평균: {np.mean(X[:, i]):.3f}, 표준편차: {np.std(X[:, i]):.3f}')
                ax.set_xlabel('값')
                ax.set_ylabel('빈도')
                ax.grid(True, alpha=0.3)
            
            # 빈 서브플롯 제거
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    fig.delaxes(axes[row, col])
                else:
                    fig.delaxes(axes[col])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"특징 분포 그래프 저장: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib와 seaborn이 필요합니다: pip install matplotlib seaborn")
    
    @staticmethod
    def plot_correlation_matrix(X: np.ndarray, feature_names: Optional[List[str]] = None,
                              output_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (12, 10)):
        """
        특징 간 상관관계 매트릭스를 시각화합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            특징 데이터
        feature_names : Optional[List[str]]
            특징 이름 목록
        output_path : Optional[Union[str, Path]]
            저장할 경로
        figsize : Tuple[int, int]
            그래프 크기
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 상관관계 매트릭스 계산
            correlation_matrix = np.corrcoef(X.T)
            
            if feature_names is None:
                feature_names = [f'F{i+1}' for i in range(X.shape[1])]
            
            # 히트맵 그리기
            plt.figure(figsize=figsize)
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       xticklabels=feature_names,
                       yticklabels=feature_names,
                       fmt='.2f')
            
            plt.title('특징 간 상관관계 매트릭스')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"상관관계 매트릭스 저장: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib와 seaborn이 필요합니다: pip install matplotlib seaborn")
    
    @staticmethod
    def plot_class_distribution(y: np.ndarray, class_names: Optional[List[str]] = None,
                              output_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (10, 6)):
        """
        클래스 분포를 시각화합니다.
        
        Parameters:
        -----------
        y : np.ndarray
            레이블 데이터
        class_names : Optional[List[str]]
            클래스 이름 목록
        output_path : Optional[Union[str, Path]]
            저장할 경로
        figsize : Tuple[int, int]
            그래프 크기
        """
        try:
            import matplotlib.pyplot as plt
            
            unique_labels, counts = np.unique(y, return_counts=True)
            
            if class_names is None:
                class_names = [f'Class {label}' for label in unique_labels]
            
            plt.figure(figsize=figsize)
            
            # 막대 그래프
            bars = plt.bar(range(len(unique_labels)), counts, 
                          alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'][:len(unique_labels)])
            
            # 값 표시
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}\n({count/len(y)*100:.1f}%)',
                        ha='center', va='bottom')
            
            plt.xlabel('클래스')
            plt.ylabel('샘플 수')
            plt.title('클래스 분포')
            plt.xticks(range(len(unique_labels)), class_names)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"클래스 분포 그래프 저장: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib이 필요합니다: pip install matplotlib")
    
    @staticmethod
    def plot_feature_importance(importance_dict: Dict[str, float],
                              top_n: int = 15,
                              output_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (12, 8)):
        """
        특징 중요도를 시각화합니다.
        
        Parameters:
        -----------
        importance_dict : Dict[str, float]
            특징 중요도 딕셔너리
        top_n : int
            상위 N개 특징만 표시
        output_path : Optional[Union[str, Path]]
            저장할 경로
        figsize : Tuple[int, int]
            그래프 크기
        """
        try:
            import matplotlib.pyplot as plt
            
            # 상위 N개 특징 선택
            top_features = list(importance_dict.items())[:top_n]
            features, scores = zip(*top_features)
            
            plt.figure(figsize=figsize)
            
            # 수평 막대 그래프
            bars = plt.barh(range(len(features)), scores, alpha=0.7, color='steelblue')
            
            # 값 표시
            for i, (bar, score) in enumerate(zip(bars, scores)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center')
            
            plt.xlabel('중요도 점수')
            plt.ylabel('특징')
            plt.title(f'특징 중요도 (상위 {top_n}개)')
            plt.yticks(range(len(features)), features)
            plt.grid(True, alpha=0.3, axis='x')
            
            # y축 뒤집기 (중요도 높은 순으로)
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"특징 중요도 그래프 저장: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("matplotlib이 필요합니다: pip install matplotlib")


class MemoryUtils(LoggerMixin):
    """
    메모리 관리 유틸리티 클래스
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        현재 메모리 사용량을 반환합니다.
        
        Returns:
        --------
        Dict[str, float]
            메모리 사용량 정보 (MB 단위)
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),  # 물리 메모리
                'vms_mb': memory_info.vms / (1024 * 1024),  # 가상 메모리
                'percent': process.memory_percent(),  # 전체 메모리 대비 사용률
            }
        except ImportError:
            return {'error': 'psutil 라이브러리가 필요합니다'}
    
    @staticmethod
    def estimate_array_memory(shape: Tuple[int, ...], dtype: np.dtype) -> float:
        """
        배열의 메모리 사용량을 추정합니다.
        
        Parameters:
        -----------
        shape : Tuple[int, ...]
            배열 모양
        dtype : np.dtype
            데이터 타입
            
        Returns:
        --------
        float
            예상 메모리 사용량 (MB)
        """
        total_elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = total_elements * bytes_per_element
        
        return total_bytes / (1024 * 1024)  # MB 변환
    
    @staticmethod
    def chunk_array(X: np.ndarray, max_memory_mb: float = 100) -> List[np.ndarray]:
        """
        배열을 메모리 제한에 맞춰 청크로 분할합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            분할할 배열
        max_memory_mb : float
            최대 메모리 사용량 (MB)
            
        Returns:
        --------
        List[np.ndarray]
            분할된 배열 청크 리스트
        """
        current_memory = MemoryUtils.estimate_array_memory(X.shape, X.dtype)
        
        if current_memory <= max_memory_mb:
            return [X]
        
        # 필요한 청크 수 계산
        n_chunks = int(np.ceil(current_memory / max_memory_mb))
        chunk_size = len(X) // n_chunks
        
        chunks = []
        for i in range(0, len(X), chunk_size):
            end_idx = min(i + chunk_size, len(X))
            chunks.append(X[i:end_idx])
        
        return chunks
    
    @staticmethod
    def memory_efficient_operation(X: np.ndarray, operation: callable,
                                 max_memory_mb: float = 100,
                                 **kwargs) -> np.ndarray:
        """
        메모리 효율적으로 연산을 수행합니다.
        
        Parameters:
        -----------
        X : np.ndarray
            입력 배열
        operation : callable
            수행할 연산 함수
        max_memory_mb : float
            최대 메모리 사용량 (MB)
        **kwargs
            연산 함수에 전달할 추가 인자
            
        Returns:
        --------
        np.ndarray
            연산 결과
        """
        chunks = MemoryUtils.chunk_array(X, max_memory_mb)
        
        if len(chunks) == 1:
            return operation(X, **kwargs)
        
        results = []
        for chunk in chunks:
            result = operation(chunk, **kwargs)
            results.append(result)
        
        return np.concatenate(results, axis=0)
    
    @staticmethod
    def clear_memory():
        """
        가비지 컬렉션을 수행하여 메모리를 정리합니다.
        """
        import gc
        
        collected = gc.collect()
        return {'collected_objects': collected}
    
    @staticmethod
    def monitor_memory_usage(func: callable):
        """
        함수 실행 중 메모리 사용량을 모니터링하는 데코레이터.
        
        Parameters:
        -----------
        func : callable
            모니터링할 함수
            
        Returns:
        --------
        callable
            래핑된 함수
        """
        def wrapper(*args, **kwargs):
            before = MemoryUtils.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                after = MemoryUtils.get_memory_usage()
                
                if 'rss_mb' in before and 'rss_mb' in after:
                    memory_diff = after['rss_mb'] - before['rss_mb']
                    print(f"함수 '{func.__name__}' 메모리 사용량: {memory_diff:+.2f} MB")
                
                return result
                
            except Exception as e:
                after = MemoryUtils.get_memory_usage()
                if 'rss_mb' in before and 'rss_mb' in after:
                    memory_diff = after['rss_mb'] - before['rss_mb']
                    print(f"함수 '{func.__name__}' 실행 중 오류 (메모리 사용량: {memory_diff:+.2f} MB)")
                raise
        
        return wrapper