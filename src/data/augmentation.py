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
    
    단일 오디오 파일에 대한 증강 프로세스의 결과를 추적합니다.
    증강 프로세스의 성공 여부와 생성된 파일들의 정보를 포함합니다.
    
    Attributes:
        original_file : str
            증강의 대상이 된 원본 오디오 파일의 경로
        augmented_files : List[str]
            생성된 모든 증강 파일들의 경로 리스트
        noise_types_used : List[str]
            증강에 사용된 소음 유형들 (예: 'homeplus', 'emart')
        snr_levels_used : List[float]
            적용된 SNR 레벨들 (dB 단위)
        total_created : int
            생성된 증강 파일의 총 개수
        skipped_noise_files : List[str]
            오류로 인해 사용할 수 없었던 소음 파일들의 경로
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
        # NaN 또는 무한대 검사
        if not np.isfinite(audio).all():
            raise ValueError("오디오 신호에 NaN 또는 무한대 값이 포함되어 있습니다.")
        
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
        # 길이 검증
        if len(signal) != len(noise):
            raise ValueError(f"신호와 소음의 길이가 다릅니다: {len(signal)} vs {len(noise)}")
        
        signal_rms = self.calculate_rms(signal)
        noise_rms = self.calculate_rms(noise)
        
        # 제로 신호 처리
        if signal_rms == 0:
            raise ValueError("신호의 RMS가 0입니다. SNR을 계산할 수 없습니다.")
        
        if noise_rms == 0:
            return 120.0  # 실용적인 최대값 (무한대 대신)
        
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
        # 길이 검증
        if len(signal) != len(noise):
            raise ValueError(f"신호와 소음의 길이가 다릅니다: {len(signal)} vs {len(noise)}")
        
        signal_rms = self.calculate_rms(signal)
        noise_rms = self.calculate_rms(noise)
        
        if noise_rms == 0:
            self.logger.warning("소음 RMS가 0입니다. 제로 배열을 반환합니다.")
            return np.zeros_like(noise)
        
        # design.md의 공식 적용
        scaling_factor = (signal_rms / noise_rms) * (10 ** (-target_snr_db / 20))
        scaled_noise = noise * scaling_factor
        
        # 극단적인 SNR 값에 대한 클리핑 방지
        max_scale = 10.0  # 최대 스케일링 팩터를 더 작게 설정
        if scaling_factor > max_scale:
            scaled_noise = noise * max_scale
            self.logger.warning(f"스케일링 팩터가 제한됨: {scaling_factor:.2f} -> {max_scale}")
        
        # 추가 클리핑 방지: 결과가 1.0을 넘지 않도록 정규화
        max_val = np.max(np.abs(scaled_noise))
        if max_val > 1.0:
            scaled_noise = scaled_noise / max_val * 0.95
        
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
    
    def mix_signals(self, signal: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        신호와 소음을 단순 혼합합니다 (SNR 조정 없이).
        
        Parameters:
        -----------
        signal : np.ndarray
            신호 오디오 데이터
        noise : np.ndarray
            소음 오디오 데이터
            
        Returns:
        --------
        np.ndarray
            혼합된 오디오 데이터
        """
        # 길이 검증
        if len(signal) != len(noise):
            raise ValueError(f"신호와 소음의 길이가 다릅니다: {len(signal)} vs {len(noise)}")
        
        # 단순 혼합
        mixed = signal + noise
        
        # 클리핑 방지를 위한 정규화
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95
        
        return mixed
    
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
    
    def calculate_dynamic_augmentation(self, num_noise_files: int, 
                                       snr_levels: List[float] = None, 
                                       base_factor: int = None) -> int:
        """
        사용 가능한 소음 파일 수에 따라 동적으로 증강 배수를 계산합니다.
        
        Parameters:
        -----------
        num_noise_files : int
            사용 가능한 소음 파일 수
        snr_levels : List[float], optional
            SNR 레벨 리스트. None이면 기본 설정 사용
        base_factor : int, optional
            기본 증강 배수. None이면 기본 설정 사용
            
        Returns:
        --------
        int
            계산된 증강 배수
        """
        if num_noise_files == 0:
            return 1  # 소음이 없으면 1 반환
        
        if snr_levels is None:
            snr_levels = self.config.snr_levels
        if base_factor is None:
            base_factor = self.config.augmentation_factor
            
        # 증강 배수 = base_factor * 소음파일수 * SNR레벨수
        factor = base_factor * num_noise_files * len(snr_levels)
        
        return factor
    
    def augment_class_directory(self, audio_dir: str, noise_dir: str = None, 
                               output_dir: str = None, snr_levels: List[float] = None,
                               augmentation_factor: int = None, class_name: str = None, 
                               noise_files: List[str] = None, output_base_dir: str = None) -> List[AugmentationResult]:
        """
        클래스별 디렉토리의 모든 오디오 파일을 증강합니다.
        
        Parameters:
        -----------
        audio_dir : str
            오디오 파일이 있는 디렉토리 경로
        noise_dir : str, optional
            소음 파일이 있는 디렉토리 경로
        output_dir : str, optional
            출력 디렉토리 경로
        snr_levels : List[float], optional
            사용할 SNR 레벨들
        augmentation_factor : int, optional
            증강 배수
        class_name : str, optional
            클래스 이름 (레거시 지원)
        noise_files : List[str], optional
            사용할 소음 파일 리스트 (레거시 지원)
        output_base_dir : str, optional
            출력 기본 디렉토리 (레거시 지원)
            
        Returns:
        --------
        List[AugmentationResult]
            각 오디오 파일에 대한 증강 결과 리스트
        """
        # 레거시 파라미터 처리
        if class_name and noise_files and output_base_dir:
            # 구 버전 호출 방식
            audio_dir = audio_dir  # class_dir로 전달됨
            output_dir = os.path.join(output_base_dir, class_name)
        
        # 기본값 설정
        if snr_levels is None:
            snr_levels = self.config.snr_levels
        
        # 출력 디렉토리 생성
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 오디오 파일 찾기
        audio_files = []
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(audio_dir, file))
        
        if not audio_files:
            self.logger.warning(f"오디오 파일이 없음: {audio_dir}")
            return []
        
        # 소음 파일 찾기
        if noise_files is None and noise_dir:
            noise_files = []
            if os.path.exists(noise_dir):
                for root, dirs, files in os.walk(noise_dir):
                    for file in files:
                        if file.lower().endswith('.wav'):
                            noise_files.append(os.path.join(root, file))
        
        if not noise_files:
            noise_files = []
        
        results = []
        
        # 각 오디오 파일에 대해 증강 수행
        for audio_file in audio_files:
            augmented_files = []
            noise_types_used = []
            snr_levels_used = []
            skipped_noise = []
            
            try:
                # 각 소음 파일과 SNR 레벨에 대해 증강
                for noise_file in noise_files:
                    for snr in snr_levels:
                        try:
                            result_file = augment_noise(
                                audio_file, noise_file, snr, output_dir or audio_dir
                            )
                            if result_file:
                                augmented_files.append(result_file)
                                noise_type = self._extract_noise_type(noise_file)
                                if noise_type not in noise_types_used:
                                    noise_types_used.append(noise_type)
                                if snr not in snr_levels_used:
                                    snr_levels_used.append(snr)
                        except Exception as e:
                            self.logger.error(f"증강 실패: {e}")
                            if noise_file not in skipped_noise:
                                skipped_noise.append(noise_file)
                
            except Exception as e:
                self.logger.error(f"파일 증강 실패 {audio_file}: {e}")
            
            result = AugmentationResult(
                original_file=audio_file,
                augmented_files=augmented_files,
                noise_types_used=noise_types_used,
                snr_levels_used=snr_levels_used,
                total_created=len(augmented_files),
                skipped_noise_files=skipped_noise
            )
            results.append(result)
        
        total_created = sum(r.total_created for r in results)
        self.logger.info(f"증강 완료: {len(audio_files)}개 파일, {total_created}개 증강 파일 생성")
        return results
    
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
    
    def validate_augmented_audio(self, audio_file: str) -> bool:
        """
        증강된 오디오 파일의 품질을 검증합니다.
        
        Parameters:
        -----------
        audio_file : str
            검증할 오디오 파일 경로
            
        Returns:
        --------
        bool
            품질이 acceptable하면 True
        """
        try:
            # 파일 로드
            audio, sr = librosa.load(audio_file, sr=None, mono=True)
            
            # 기본 검증
            if len(audio) == 0:
                return False
            
            # NaN 또는 무한대 값 검사
            if not np.isfinite(audio).all():
                self.logger.warning(f"오디오에 NaN 또는 무한대 값 포함: {audio_file}")
                return False
            
            # RMS 검사 (너무 조용하지 않은지)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 1e-6:
                self.logger.warning(f"오디오가 너무 조용함: {audio_file}")
                return False
            
            # 클리핑 검사
            if np.max(np.abs(audio)) > 1.0:
                self.logger.warning(f"오디오 클리핑 감지: {audio_file}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"오디오 검증 실패 {audio_file}: {e}")
            return False
    
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


def augment_noise(clean_audio_path: str, noise_files, 
                  snr_levels, output_dir: str, 
                  config=None):
    """
    깨끗한 오디오의 소음 증강 버전을 생성합니다.
    
    design.md에 명시된 인터페이스를 구현합니다.
    
    Parameters:
    -----------
    clean_audio_path : str
        원본 깨끗한 오디오 파일의 경로
    noise_files : str or List[str]
        소음 오디오 파일 경로 (단일 파일 또는 목록)
    snr_levels : float or List[float]
        소음 혼합을 위한 SNR 값(dB) (단일 값 또는 목록)
    output_dir : str
        증강된 오디오 파일을 저장할 디렉토리
    config : Config, optional
        구성 객체
        
    Returns:
    --------
    str or List[str]
        생성된 증강 오디오 파일의 경로(들)
    """
    augmentor = AudioAugmentor(config)
    logger = augmentor.logger
    
    # 파라미터 정규화: 단일 값을 리스트로 변환
    if isinstance(noise_files, str):
        noise_files = [noise_files]
        single_noise = True
    else:
        single_noise = False
    
    if isinstance(snr_levels, (int, float)):
        snr_levels = [snr_levels]
        single_snr = True
    else:
        single_snr = False
    
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
    
    # 단일 파일 요청인 경우 단일 결과 반환
    if single_noise and single_snr and len(augmented_files) == 1:
        return augmented_files[0]
    
    return augmented_files