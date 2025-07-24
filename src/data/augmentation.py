"""
데이터 증강 모듈

이 모듈은 SNR 제어를 통한 소음 증강 기능을 제공합니다.
design.md에 명시된 인터페이스와 수학적 기초를 구현합니다.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from ..utils.logger import LoggerMixin
from config import DEFAULT_CONFIG


@dataclass
class AugmentationResult:
    """
    증강 결과를 나타내는 데이터 클래스.
    """
    original_file: str
    augmented_files: List[str]
    noise_types_used: List[str]
    snr_levels_used: List[float]
    total_created: int
    skipped_noise_files: List[str]  # 사용할 수 없었던 소음 파일들


class AudioAugmentor(LoggerMixin):
    """
    오디오 데이터 증강을 위한 클래스.
    
    SNR 제어를 통해 깨끗한 오디오와 소음을 혼합하여
    강건한 훈련 데이터를 생성합니다.
    """
    
    def __init__(self, config=None):
        """
        증강기를 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        self.logger.info(f"AudioAugmentor 초기화됨 - SNR 레벨: {self.config.snr_levels}, "
                        f"증강 배수: {self.config.augmentation_factor}")
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """
        오디오 신호의 RMS (Root Mean Square) 에너지를 계산합니다.
        
        Parameters:
        -----------
        audio : np.ndarray
            오디오 시계열 데이터
            
        Returns:
        --------
        float
            RMS 에너지 값
        """
        return np.sqrt(np.mean(audio ** 2))
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        신호와 소음의 SNR을 계산합니다.
        
        Parameters:
        -----------
        signal : np.ndarray
            신호 오디오 데이터
        noise : np.ndarray
            소음 오디오 데이터
            
        Returns:
        --------
        float
            SNR 값 (dB)
        """
        signal_rms = self.calculate_rms(signal)
        noise_rms = self.calculate_rms(noise)
        
        if noise_rms == 0:
            return float('inf')  # 소음이 없는 경우
        
        snr_linear = signal_rms / noise_rms
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')
        
        return snr_db
    
    def scale_noise_for_snr(self, signal: np.ndarray, noise: np.ndarray, 
                           target_snr_db: float) -> np.ndarray:
        """
        목표 SNR을 달성하기 위해 소음을 스케일링합니다.
        
        design.md의 수학적 기초를 구현:
        noise_scaled = noise * (RMS_signal / RMS_noise) * 10^(-SNR_dB/20)
        
        Parameters:
        -----------
        signal : np.ndarray
            신호 오디오 데이터
        noise : np.ndarray
            소음 오디오 데이터
        target_snr_db : float
            목표 SNR 값 (dB)
            
        Returns:
        --------
        np.ndarray
            스케일링된 소음 데이터
        """
        signal_rms = self.calculate_rms(signal)
        noise_rms = self.calculate_rms(noise)
        
        if noise_rms == 0:
            self.logger.warning("소음 RMS가 0입니다. 원본 소음을 반환합니다.")
            return noise
        
        # design.md의 공식 적용
        scaling_factor = (signal_rms / noise_rms) * (10 ** (-target_snr_db / 20))
        scaled_noise = noise * scaling_factor
        
        return scaled_noise
    
    def load_and_match_length(self, audio_path: str, target_length: int, 
                             sr: int) -> Optional[np.ndarray]:
        """
        오디오 파일을 로드하고 목표 길이에 맞춥니다.
        
        Parameters:
        -----------
        audio_path : str
            오디오 파일 경로
        target_length : int
            목표 오디오 길이 (샘플 수)
        sr : int
            샘플링 레이트
            
        Returns:
        --------
        np.ndarray or None
            길이가 조정된 오디오 데이터, 실패 시 None
        """
        try:
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            if len(audio) == 0:
                self.logger.warning(f"빈 오디오 파일: {audio_path}")
                return None
            
            # 길이 조정
            if len(audio) >= target_length:
                # 소음이 더 긴 경우: 무작위 위치에서 잘라내기
                start_idx = random.randint(0, len(audio) - target_length)
                audio = audio[start_idx:start_idx + target_length]
            else:
                # 소음이 더 짧은 경우: 반복해서 목표 길이 채우기
                repeat_times = (target_length // len(audio)) + 1
                audio = np.tile(audio, repeat_times)[:target_length]
            
            return audio
            
        except Exception as e:
            self.logger.error(f"오디오 로드 실패 {audio_path}: {e}")
            return None
    
    def mix_audio_with_noise(self, signal: np.ndarray, noise: np.ndarray, 
                           snr_db: float) -> np.ndarray:
        """
        신호와 소음을 지정된 SNR로 혼합합니다.
        
        Parameters:
        -----------
        signal : np.ndarray
            신호 오디오 데이터
        noise : np.ndarray
            소음 오디오 데이터
        snr_db : float
            목표 SNR 값 (dB)
            
        Returns:
        --------
        np.ndarray
            혼합된 오디오 데이터
        """
        # 소음을 목표 SNR에 맞게 스케일링
        scaled_noise = self.scale_noise_for_snr(signal, noise, snr_db)
        
        # 신호와 소음 혼합
        mixed_audio = signal + scaled_noise
        
        # 클리핑 방지를 위한 정규화
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val * 0.95  # 약간의 여유 공간 두기
        
        return mixed_audio
    
    def generate_augmented_filename(self, original_path: str, noise_file: str, 
                                  snr_db: float, output_dir: str) -> str:
        """
        증강된 파일의 이름을 생성합니다.
        
        Parameters:
        -----------
        original_path : str
            원본 파일 경로
        noise_file : str
            사용된 소음 파일 경로
        snr_db : float
            사용된 SNR 값
        output_dir : str
            출력 디렉토리
            
        Returns:
        --------
        str
            생성된 파일 경로
        """
        original_name = Path(original_path).stem
        noise_name = Path(noise_file).stem
        
        # 파일명 형식: 원본명_noise_소음명_snr값dB.wav
        filename = f"{original_name}_noise_{noise_name}_snr{snr_db:+.0f}dB.wav"
        
        return os.path.join(output_dir, filename)
    
    def validate_augmented_audio(self, audio: np.ndarray, 
                               original_audio: np.ndarray) -> bool:
        """
        증강된 오디오의 품질을 검증합니다.
        
        Parameters:
        -----------
        audio : np.ndarray
            증강된 오디오 데이터
        original_audio : np.ndarray
            원본 오디오 데이터
            
        Returns:
        --------
        bool
            품질이 acceptable하면 True
        """
        # 기본 검증
        if len(audio) == 0:
            return False
        
        # NaN 또는 무한대 값 검사
        if not np.isfinite(audio).all():
            self.logger.warning("증강된 오디오에 NaN 또는 무한대 값 포함")
            return False
        
        # 동적 범위 검사 (너무 조용하거나 클리핑되지 않았는지)
        audio_rms = self.calculate_rms(audio)
        original_rms = self.calculate_rms(original_audio)
        
        # RMS가 원본의 10%보다 작거나 10배보다 크면 품질 문제
        if audio_rms < original_rms * 0.1:
            self.logger.warning("증강된 오디오가 너무 조용함")
            return False
        
        if audio_rms > original_rms * 10:
            self.logger.warning("증강된 오디오가 너무 큼")
            return False
        
        # 클리핑 검사
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        if clipping_ratio > 0.01:  # 1% 이상 클리핑되면 문제
            self.logger.warning(f"증강된 오디오에 과도한 클리핑: {clipping_ratio:.2%}")
            return False
        
        return True


class BatchAugmentor(LoggerMixin):
    """
    배치 처리를 위한 데이터 증강 클래스.
    
    여러 파일을 효율적으로 처리하고 동적 증강 배수를 지원합니다.
    """
    
    def __init__(self, config=None):
        """
        배치 증강기를 초기화합니다.
        
        Parameters:
        -----------
        config : Config, optional
            구성 객체. None이면 기본 구성을 사용합니다.
        """
        self.config = config or DEFAULT_CONFIG
        self.augmentor = AudioAugmentor(config)
        self.logger.info(f"BatchAugmentor 초기화됨")
    
    def calculate_dynamic_augmentation(self, available_noise_files: int) -> Tuple[List[float], int]:
        """
        사용 가능한 소음 파일 수에 따라 동적으로 증강 전략을 조정합니다.
        
        Parameters:
        -----------
        available_noise_files : int
            사용 가능한 소음 파일 수
            
        Returns:
        --------
        Tuple[List[float], int]
            (조정된 SNR 레벨들, 실제 증강 배수)
        """
        if available_noise_files < self.config.min_noise_files:
            self.logger.warning(f"소음 파일 부족: {available_noise_files}개 (최소 {self.config.min_noise_files}개 필요)")
            return self.config.snr_levels[:1], 1  # 최소한의 증강만 수행
        
        # 소음 파일 수에 따른 SNR 레벨 조정
        snr_levels = self.config.snr_levels.copy()
        
        if available_noise_files >= len(self.config.snr_levels):
            # 충분한 소음 파일: 모든 SNR 레벨 사용
            actual_factor = min(self.config.augmentation_factor, 
                               available_noise_files * len(self.config.snr_levels))
        else:
            # 제한된 소음 파일: SNR 레벨 수 조정
            snr_per_noise = max(1, len(self.config.snr_levels) // available_noise_files)
            snr_levels = self.config.snr_levels[:snr_per_noise]
            actual_factor = available_noise_files * len(snr_levels)
        
        self.logger.info(f"동적 증강 설정: {len(snr_levels)}개 SNR 레벨, "
                        f"실제 배수: {actual_factor}")
        
        return snr_levels, actual_factor
    
    def augment_class_directory(self, class_dir: str, class_name: str, 
                              noise_files: List[str], output_base_dir: str) -> AugmentationResult:
        """
        클래스별 디렉토리의 모든 오디오 파일을 증강합니다.
        
        Parameters:
        -----------
        class_dir : str
            클래스 디렉토리 경로
        class_name : str
            클래스 이름 (watermelon_A, watermelon_B, watermelon_C)
        noise_files : List[str]
            사용 가능한 소음 파일 목록
        output_base_dir : str
            출력 기본 디렉토리
            
        Returns:
        --------
        AugmentationResult
            증강 결과
        """
        self.logger.info(f"클래스 증강 시작: {class_name}")
        
        # 출력 디렉토리 생성
        output_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 클래스 디렉토리의 모든 WAV 파일 찾기
        audio_files = []
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(class_dir, file))
        
        if not audio_files:
            self.logger.warning(f"오디오 파일이 없음: {class_dir}")
            return AugmentationResult(
                original_file=class_dir,
                augmented_files=[],
                noise_types_used=[],
                snr_levels_used=[],
                total_created=0,
                skipped_noise_files=noise_files
            )
        
        # 동적 증강 설정 계산
        snr_levels, actual_factor = self.calculate_dynamic_augmentation(len(noise_files))
        
        # 소음 파일을 증강 배수에 맞게 선택
        selected_noise_files = noise_files.copy()
        if len(noise_files) > actual_factor:
            selected_noise_files = random.sample(noise_files, actual_factor)
        
        all_augmented_files = []
        noise_types_used = []
        skipped_noise = []
        
        # 각 오디오 파일에 대해 증강 수행
        for audio_file in audio_files:
            try:
                augmented = augment_noise(
                    audio_file, selected_noise_files, snr_levels, output_dir, self.config
                )
                all_augmented_files.extend(augmented)
                
                # 성공한 소음 타입 추적
                for noise_file in selected_noise_files:
                    noise_type = self._extract_noise_type(noise_file)
                    if noise_type not in noise_types_used:
                        noise_types_used.append(noise_type)
                
            except Exception as e:
                self.logger.error(f"파일 증강 실패 {audio_file}: {e}")
                continue
        
        result = AugmentationResult(
            original_file=class_dir,
            augmented_files=all_augmented_files,
            noise_types_used=noise_types_used,
            snr_levels_used=snr_levels,
            total_created=len(all_augmented_files),
            skipped_noise_files=skipped_noise
        )
        
        self.logger.info(f"클래스 증강 완료: {class_name}, {result.total_created}개 파일 생성")
        return result
    
    def _extract_noise_type(self, noise_file_path: str) -> str:
        """
        소음 파일 경로에서 소음 타입을 추출합니다.
        
        Parameters:
        -----------
        noise_file_path : str
            소음 파일 경로
            
        Returns:
        --------
        str
            소음 타입 (예: "homeplus", "emart", "mechanical")
        """
        path_parts = Path(noise_file_path).parts
        
        # 경로에서 소음 유형 추출 시도
        for part in reversed(path_parts):
            if part in ['homeplus', 'emart', 'mechanical', 'background']:
                return part
        
        # 파일명에서 추출 시도
        filename = Path(noise_file_path).stem.lower()
        if 'homeplus' in filename:
            return 'homeplus'
        elif 'emart' in filename:
            return 'emart'
        elif 'mechanical' in filename:
            return 'mechanical'
        elif 'background' in filename:
            return 'background'
        
        return 'unknown'
    
    def cleanup_augmented_files(self, augmented_files: List[str]) -> int:
        """
        임시 증강 파일들을 정리합니다.
        
        Parameters:
        -----------
        augmented_files : List[str]
            삭제할 증강 파일 경로들
            
        Returns:
        --------
        int
            삭제된 파일 수
        """
        deleted_count = 0
        
        for file_path in augmented_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    self.logger.debug(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                self.logger.error(f"파일 삭제 실패 {file_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"임시 증강 파일 정리 완료: {deleted_count}개 파일 삭제")
        
        return deleted_count


def augment_noise(clean_audio_path: str, noise_files: List[str], 
                  snr_levels: List[float], output_dir: str, 
                  config=None) -> List[str]:
    """
    깨끗한 오디오의 소음 증강 버전을 생성합니다.
    
    design.md에 명시된 인터페이스를 구현합니다.
    
    Parameters:
    -----------
    clean_audio_path : str
        원본 깨끗한 오디오 파일의 경로
    noise_files : List[str]
        소음 오디오 파일 경로 목록
    snr_levels : List[float]
        소음 혼합을 위한 SNR 값(dB)
    output_dir : str
        증강된 오디오 파일을 저장할 디렉토리
    config : Config, optional
        구성 객체
        
    Returns:
    --------
    List[str]
        생성된 증강 오디오 파일의 경로들
    """
    augmentor = AudioAugmentor(config)
    logger = augmentor.logger
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"소음 증강 시작: {clean_audio_path}")
    logger.info(f"사용 가능한 소음 파일: {len(noise_files)}개, SNR 레벨: {snr_levels}")
    
    # 원본 오디오 로드
    try:
        clean_audio, sr = librosa.load(clean_audio_path, 
                                     sr=augmentor.config.sample_rate, 
                                     mono=True)
        if len(clean_audio) == 0:
            logger.error(f"빈 원본 오디오 파일: {clean_audio_path}")
            return []
            
    except Exception as e:
        logger.error(f"원본 오디오 로드 실패 {clean_audio_path}: {e}")
        return []
    
    augmented_files = []
    skipped_noise = []
    
    # 각 소음 파일과 SNR 레벨 조합으로 증강
    for noise_file in noise_files:
        # 소음 파일 로드 및 길이 조정
        noise_audio = augmentor.load_and_match_length(
            noise_file, len(clean_audio), sr
        )
        
        if noise_audio is None:
            skipped_noise.append(noise_file)
            continue
        
        for snr_db in snr_levels:
            try:
                # 오디오 혼합
                mixed_audio = augmentor.mix_audio_with_noise(
                    clean_audio, noise_audio, snr_db
                )
                
                # 품질 검증
                if not augmentor.validate_augmented_audio(mixed_audio, clean_audio):
                    logger.warning(f"품질 검증 실패: {noise_file}, SNR={snr_db}dB")
                    continue
                
                # 파일명 생성 및 저장
                output_path = augmentor.generate_augmented_filename(
                    clean_audio_path, noise_file, snr_db, output_dir
                )
                
                sf.write(output_path, mixed_audio, sr)
                augmented_files.append(output_path)
                
                logger.debug(f"증강 파일 생성: {os.path.basename(output_path)}")
                
            except Exception as e:
                logger.error(f"증강 실패 {noise_file}, SNR={snr_db}dB: {e}")
                continue
    
    logger.info(f"소음 증강 완료: {len(augmented_files)}개 파일 생성, "
               f"{len(skipped_noise)}개 소음 파일 건너뜀")
    
    return augmented_files