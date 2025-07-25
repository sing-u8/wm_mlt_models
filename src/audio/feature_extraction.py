"""
오디오 특징 추출 모듈

이 모듈은 librosa를 사용하여 수박 소리에서 포괄적인 오디오 특징을 추출합니다.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import librosa
import librosa.feature
from pathlib import Path

from ..utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class FeatureVector:
    """
    오디오 특징 벡터를 나타내는 데이터 클래스.
    
    design.md에 명시된 구조를 따릅니다. 이 클래스는 추출된 모든 오디오 특징을
    구조화된 형태로 저장하고 ML 모델 입력용 평면 배열로 변환하는 기능을 제공합니다.
    
    Attributes:
        mfcc : np.ndarray
            13개 MFCC 계수의 평균값. 형태: (13,)
        mel_mean : float
            멜 스펙트로그램의 평균값 (스칼라)
        mel_std : float
            멜 스펙트로그램의 표준편차 (스칼라)
        spectral_centroid : float
            스펙트럴 중심의 평균값 (스칼라)
        spectral_rolloff : float
            스펙트럴 롤오프의 평균값 (스칼라)
        zero_crossing_rate : float
            제로 교차율의 평균값 (스칼라)
        chroma : np.ndarray
            12개 크로마 특징의 평균값. 형태: (12,)
    """
    mfcc: np.ndarray          # 형태: (13,)
    mel_mean: float           # 스칼라
    mel_std: float            # 스칼라  
    spectral_centroid: float  # 스칼라
    spectral_rolloff: float   # 스칼라
    zero_crossing_rate: float # 스칼라
    chroma: np.ndarray        # 형태: (12,)
    
    def to_array(self) -> np.ndarray:
        """
        ML 모델을 위해 평면 numpy 배열로 변환합니다.
        
        Returns:
            np.ndarray: 모든 특징이 연결된 1차원 배열. 형태: (30,)
                       순서: MFCC(13) + 통계(5) + Chroma(12)
        """
        return np.concatenate([
            self.mfcc,
            [self.mel_mean, self.mel_std, self.spectral_centroid, 
             self.spectral_rolloff, self.zero_crossing_rate],
            self.chroma
        ])  # 총 형태: (30,)
    
    @property
    def feature_names(self) -> list:
        """
        특징 이름 목록을 반환합니다.
        
        Returns:
            list: 30개 특징의 이름 리스트
                 예: ['mfcc_1', 'mfcc_2', ..., 'mel_mean', ..., 'chroma_1', ...]
        """
        names = []
        
        # MFCC 특징 이름
        for i in range(len(self.mfcc)):
            names.append(f'mfcc_{i+1}')
        
        # 기타 특징 이름
        names.extend([
            'mel_mean', 'mel_std', 'spectral_centroid',
            'spectral_rolloff', 'zero_crossing_rate'
        ])
        
        # Chroma 특징 이름
        for i in range(len(self.chroma)):
            names.append(f'chroma_{i+1}')
        
        return names


class AudioFeatureExtractor(LoggerMixin):
    """
    오디오 특징 추출기 클래스.
    
    librosa를 사용하여 다양한 오디오 특징을 추출합니다.
    """
    
    def __init__(self, config=None):
        """
        특징 추출기를 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        self.logger.info(f"AudioFeatureExtractor 초기화됨 - SR: {self.config.sample_rate}, "
                        f"Hop: {self.config.hop_length}, MFCC: {self.config.n_mfcc}, "
                        f"Chroma: {self.config.n_chroma}")
    
    def validate_audio_file(self, audio_file_path: str) -> bool:
        """
        오디오 파일의 유효성을 검사합니다.
        
        Parameters:
        -----------
        audio_file_path : str
            검사할 오디오 파일 경로
            
        Returns:
        --------
        bool
            파일이 유효하면 True, 그렇지 않으면 False
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(audio_file_path):
                self.logger.error(f"파일을 찾을 수 없음: {audio_file_path}")
                return False
            
            # 파일 확장자 확인
            file_ext = Path(audio_file_path).suffix.lower()
            if file_ext not in ['.wav', '.mp3', '.flac', '.m4a']:
                self.logger.warning(f"지원되지 않는 오디오 형식: {file_ext}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                self.logger.error(f"빈 파일: {audio_file_path}")
                return False
            
            # librosa로 파일 로드 테스트
            y, sr = librosa.load(audio_file_path, sr=None, duration=0.1)  # 처음 0.1초만 테스트
            
            if len(y) == 0:
                self.logger.error(f"오디오 데이터가 없음: {audio_file_path}")
                return False
            
            # 무음 감지 (RMS 에너지 기반)
            rms = librosa.feature.rms(y=y)[0]
            if np.mean(rms) < 1e-6:  # 매우 낮은 에너지 임계값
                self.logger.warning(f"무음 또는 매우 낮은 에너지 감지: {audio_file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"오디오 파일 검증 실패 {audio_file_path}: {e}")
            return False
    
    def load_audio(self, audio_file_path: str) -> tuple:
        """
        오디오 파일을 로드합니다.
        
        Parameters:
        -----------
        audio_file_path : str
            로드할 오디오 파일 경로
            
        Returns:
        --------
        tuple
            (audio_data, sample_rate) 또는 (None, None) if failed
        """
        try:
            # 일관된 샘플링 레이트로 로드
            y, sr = librosa.load(
                audio_file_path, 
                sr=self.config.sample_rate,
                mono=True  # 모노로 변환
            )
            
            if len(y) == 0:
                self.logger.error(f"빈 오디오 데이터: {audio_file_path}")
                return None, None
            
            self.logger.debug(f"오디오 로드 성공: {audio_file_path} "
                            f"(길이: {len(y)}, SR: {sr})")
            
            return y, sr
            
        except Exception as e:
            self.logger.error(f"오디오 로드 실패 {audio_file_path}: {e}")
            return None, None
    
    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) 특징을 추출합니다.
        
        Parameters:
        -----------
        y : np.ndarray
            오디오 시계열 데이터
        sr : int
            샘플링 레이트
            
        Returns:
        --------
        np.ndarray
            MFCC 특징 (13차원)
        """
        try:
            # MFCC 추출
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmax=self.config.fmax
            )
            
            # 시간 축에 대한 평균을 계산하여 정적 특징으로 변환
            mfcc_features = np.mean(mfccs, axis=1)
            
            return mfcc_features
            
        except Exception as e:
            self.logger.error(f"MFCC 추출 실패: {e}")
            return np.zeros(self.config.n_mfcc)
    
    def extract_mel_spectrogram_stats(self, y: np.ndarray, sr: int) -> tuple:
        """
        Mel Spectrogram의 통계적 특징을 추출합니다.
        
        Parameters:
        -----------
        y : np.ndarray
            오디오 시계열 데이터
        sr : int
            샘플링 레이트
            
        Returns:
        --------
        tuple
            (mel_mean, mel_std)
        """
        try:
            # Mel Spectrogram 계산
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmax=self.config.fmax
            )
            
            # 로그 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 통계 계산
            mel_mean = np.mean(mel_spec_db)
            mel_std = np.std(mel_spec_db)
            
            return mel_mean, mel_std
            
        except Exception as e:
            self.logger.error(f"Mel Spectrogram 통계 추출 실패: {e}")
            return 0.0, 0.0
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> tuple:
        """
        스펙트럼 특징들을 추출합니다.
        
        Parameters:
        -----------
        y : np.ndarray
            오디오 시계열 데이터
        sr : int
            샘플링 레이트
            
        Returns:
        --------
        tuple
            (spectral_centroid, spectral_rolloff, zero_crossing_rate)
        """
        try:
            # Spectral Centroid (스펙트럼의 질량 중심)
            spec_centroid = librosa.feature.spectral_centroid(
                y=y, 
                sr=sr,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft
            )
            centroid_mean = np.mean(spec_centroid)
            
            # Spectral Rolloff (스펙트럼 에너지의 85%가 포함되는 주파수)
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=y, 
                sr=sr,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
                roll_percent=0.85
            )
            rolloff_mean = np.mean(spec_rolloff)
            
            # Zero Crossing Rate (신호의 부호 변화율)
            zcr = librosa.feature.zero_crossing_rate(
                y,
                hop_length=self.config.hop_length
            )
            zcr_mean = np.mean(zcr)
            
            return centroid_mean, rolloff_mean, zcr_mean
            
        except Exception as e:
            self.logger.error(f"스펙트럼 특징 추출 실패: {e}")
            return 0.0, 0.0, 0.0
    
    def extract_chroma_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Chroma Features (12차원 피치 클래스 프로파일)를 추출합니다.
        
        Parameters:
        -----------
        y : np.ndarray
            오디오 시계열 데이터
        sr : int
            샘플링 레이트
            
        Returns:
        --------
        np.ndarray
            Chroma 특징 (12차원)
        """
        try:
            # Chroma 특징 추출
            chroma = librosa.feature.chroma_stft(
                y=y, 
                sr=sr,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
                n_chroma=self.config.n_chroma
            )
            
            # 시간 축에 대한 평균을 계산
            chroma_features = np.mean(chroma, axis=1)
            
            return chroma_features
            
        except Exception as e:
            self.logger.error(f"Chroma 특징 추출 실패: {e}")
            return np.zeros(self.config.n_chroma)


def extract_features(audio_file_path: str, config=None) -> Optional[FeatureVector]:
    """
    단일 오디오 파일에서 포괄적인 오디오 특징을 추출합니다.
    
    이 함수는 design.md에 명시된 인터페이스를 구현합니다.
    
    Parameters:
    -----------
    audio_file_path : str
        .wav 오디오 파일의 경로
    config : Config, optional
        구성 객체. None이면 기본 구성을 사용합니다.
        
    Returns:
    --------
    FeatureVector or None
        추출된 모든 특징을 포함하는 특징 벡터.
        추출 실패 시 None 반환.
    """
    extractor = AudioFeatureExtractor(config)
    logger = extractor.logger
    
    # 파일 유효성 검사
    if not extractor.validate_audio_file(audio_file_path):
        return None
    
    # 오디오 로드
    y, sr = extractor.load_audio(audio_file_path)
    if y is None:
        return None
    
    try:
        logger.debug(f"특징 추출 시작: {audio_file_path}")
        
        # 1. MFCC 특징 추출 (13개 계수)
        mfcc = extractor.extract_mfcc(y, sr)
        
        # 2. Mel Spectrogram 통계 계산
        mel_mean, mel_std = extractor.extract_mel_spectrogram_stats(y, sr)
        
        # 3. 스펙트럼 특징들 추출
        spectral_centroid, spectral_rolloff, zero_crossing_rate = \
            extractor.extract_spectral_features(y, sr)
        
        # 4. Chroma Features 추출 (12차원)
        chroma = extractor.extract_chroma_features(y, sr)
        
        # 5. FeatureVector 객체 생성
        feature_vector = FeatureVector(
            mfcc=mfcc,
            mel_mean=mel_mean,
            mel_std=mel_std,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            chroma=chroma
        )
        
        # 특징 벡터 검증
        feature_array = feature_vector.to_array()
        expected_size = extractor.config.n_mfcc + 5 + extractor.config.n_chroma  # 13 + 5 + 12 = 30
        
        if len(feature_array) != expected_size:
            logger.error(f"특징 벡터 크기 불일치: 예상 {expected_size}, 실제 {len(feature_array)}")
            return None
        
        # NaN 또는 무한대 값 검사
        if not np.isfinite(feature_array).all():
            logger.warning(f"특징 벡터에 NaN 또는 무한대 값 포함: {audio_file_path}")
            # NaN을 0으로, 무한대를 매우 큰/작은 값으로 대체
            feature_array = np.nan_to_num(feature_array, 
                                        nan=0.0, 
                                        posinf=1e6, 
                                        neginf=-1e6)
            
            # 수정된 값으로 FeatureVector 재생성
            feature_vector = FeatureVector(
                mfcc=feature_array[:extractor.config.n_mfcc],
                mel_mean=feature_array[extractor.config.n_mfcc],
                mel_std=feature_array[extractor.config.n_mfcc + 1],
                spectral_centroid=feature_array[extractor.config.n_mfcc + 2],
                spectral_rolloff=feature_array[extractor.config.n_mfcc + 3],
                zero_crossing_rate=feature_array[extractor.config.n_mfcc + 4],
                chroma=feature_array[extractor.config.n_mfcc + 5:]
            )
        
        logger.debug(f"특징 추출 완료: {audio_file_path} "
                    f"(크기: {len(feature_array)}, 평균: {np.mean(feature_array):.4f})")
        
        return feature_vector
        
    except Exception as e:
        logger.error(f"특징 추출 중 오류 발생 {audio_file_path}: {e}")
        return None