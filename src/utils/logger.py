"""
Logging infrastructure for watermelon sound classifier.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "watermelon_classifier",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    로깅 시스템 설정.
    
    Parameters:
    -----------
    name : str
        로거 이름
    log_level : str
        로그 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_dir : str, optional
        로그 파일 저장 디렉토리. None이면 프로젝트 루트의 logs/ 사용
    console_output : bool
        콘솔 출력 여부
        
    Returns:
    --------
    logging.Logger
        설정된 로거 인스턴스
    """
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 파일 핸들러 설정
    if log_dir is None:
        # 프로젝트 루트 기준으로 logs 디렉토리 설정
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 날짜별 로그 파일 생성
    log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = log_dir / log_filename
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 설정
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger '{name}' initialized. Log file: {log_filepath}")
    
    return logger


def get_logger(name: str = "watermelon_classifier") -> logging.Logger:
    """
    기존 로거 반환 또는 기본 설정으로 새 로거 생성.
    
    Parameters:
    -----------
    name : str
        로거 이름
        
    Returns:
    --------
    logging.Logger
        로거 인스턴스
    """
    logger = logging.getLogger(name)
    
    # 로거가 핸들러를 가지고 있지 않으면 기본 설정으로 초기화
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class LoggerMixin:
    """
    클래스에 로깅 기능을 추가하는 믹스인 클래스.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """클래스별 로거 반환."""
        logger_name = f"watermelon_classifier.{self.__class__.__name__}"
        return get_logger(logger_name)


# 기본 로거 인스턴스
default_logger = setup_logger()