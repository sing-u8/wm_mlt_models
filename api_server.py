from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import joblib
import numpy as np
import uvicorn
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

from src.audio.feature_extraction import extract_features, AudioFeatureExtractor
from config.config import DEFAULT_CONFIG


# JSON 직렬화 문제 해결을 위한 클래스 정의
class NumpyEncoder(json.JSONEncoder):
    """NumPy 데이터 타입을 JSON으로 직렬화하기 위한 인코더"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 로깅 설정 - 더 상세한 포맷
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="수박 당도 예측 서버", description="오디오 파일을 분석하여 수박의 당도를 예측합니다")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 배포시에는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
MODEL_PATH = os.path.join(DEFAULT_CONFIG.model_output_dir, "pickle", "random_forest_model.pkl")

try:
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        logger.info(f"모델 로드 성공: {MODEL_PATH} (딕셔너리에서 'model' 키 사용)")
    else:
        model = model_bundle
        logger.info(f"모델 로드 성공: {MODEL_PATH} (직접 모델 객체 사용)")

    # 모델 정보 로깅
    if hasattr(model, 'n_features_in_'):
        logger.info(f"모델 특성 수: {model.n_features_in_}")
    if hasattr(model, 'classes_'):
        logger.info(f"모델 클래스: {model.classes_}")
    if hasattr(model, 'n_estimators'):
        logger.info(f"Random Forest 트리 수: {model.n_estimators}")

except Exception as e:
    logger.error(f"모델 로드 실패: {e}")
    model = None


def log_audio_info(y, sr, file_path):
    """오디오 파일의 기본 정보를 로깅"""
    duration = len(y) / sr
    max_amplitude = np.max(np.abs(y))
    rms_energy = np.sqrt(np.mean(y ** 2))

    logger.info(f"📁 파일 정보: {os.path.basename(file_path)}")
    logger.info(f"   - 샘플링 레이트: {sr} Hz")
    logger.info(f"   - 길이: {duration:.2f}초 ({len(y)} 샘플)")
    logger.info(f"   - 최대 진폭: {max_amplitude:.6f}")
    logger.info(f"   - RMS 에너지: {rms_energy:.6f}")
    logger.info(f"   - 다이나믹 레인지: {20 * np.log10(max_amplitude / (rms_energy + 1e-8)):.2f} dB")


def log_feature_details(feature_vector):
    """추출된 특성의 상세 정보를 로깅"""
    feature_array = feature_vector.to_array()
    feature_names = feature_vector.feature_names

    logger.info(f"🔍 특성 분석:")
    logger.info(f"   - 특성 개수: {len(feature_array)}")
    logger.info(f"   - 값 범위: [{np.min(feature_array):.6f}, {np.max(feature_array):.6f}]")
    logger.info(f"   - 평균값: {np.mean(feature_array):.6f}")
    logger.info(f"   - 표준편차: {np.std(feature_array):.6f}")

    # MFCC 상세 정보
    logger.info(f"   - MFCC 계수 상세:")
    for i, val in enumerate(feature_vector.mfcc):
        logger.info(f"     [{i + 1:2d}] {val:10.6f}")

    # Chroma 상세 정보
    logger.info(f"   - Chroma 계수 상세:")
    for i, val in enumerate(feature_vector.chroma):
        logger.info(f"     [{i + 1:2d}] {val:10.6f}")


def save_features_to_json(feature_vector, file_path):
    """특성을 JSON 파일로 저장"""
    try:
        feature_array = feature_vector.to_array()
        feature_names = feature_vector.feature_names

        feature_data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": os.path.basename(file_path),
            "feature_count": len(feature_array),
            "features": {
                name: float(value) for name, value in zip(feature_names, feature_array)
            },
            "statistics": {
                "min": float(np.min(feature_array)),
                "max": float(np.max(feature_array)),
                "mean": float(np.mean(feature_array)),
                "std": float(np.std(feature_array)),
                "median": float(np.median(feature_array))
            }
        }

        json_filename = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(feature_data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 특성 데이터 저장됨: {json_filename}")
        return json_filename

    except Exception as e:
        logger.warning(f"특성 JSON 저장 실패: {e}")
        return None


# 서버 상태 확인
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "수박 당도 예측 서버가 실행 중입니다",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


# 지원되는 파일 형식 정보
@app.get("/supported-formats")
def get_supported_formats():
    return {
        "formats": [".wav", ".m4a", ".mp3"],
        "description": "지원되는 오디오 파일 형식"
    }


# 특성 정보 API (디버깅용)
@app.get("/feature-info")
def get_feature_info():
    return {
        "total_features": DEFAULT_CONFIG.n_mfcc + 5 + DEFAULT_CONFIG.n_chroma,  # 13 + 5 + 12 = 30
        "feature_groups": {
            "mfcc": {"count": DEFAULT_CONFIG.n_mfcc, "description": "Mel-frequency cepstral coefficients"},
            "mel_spectrogram": {"count": 2, "description": "멜 스펙트로그램의 통계 (평균, 표준편차)"},
            "spectral": {"count": 2, "description": "스펙트럼 특징 (중심, 롤오프)"},
            "zcr": {"count": 1, "description": "Zero crossing rate"},
            "chroma": {"count": DEFAULT_CONFIG.n_chroma, "description": "Chroma features"}
        },
        "model_type": model.__class__.__name__ if model else "모델 로드 실패",
        "classes": [int(c) for c in model.classes_] if hasattr(model, 'classes_') else ["알 수 없음"]
    }


# 디버깅용 특성 분석 API
@app.post("/debug-features")
async def debug_features(file: UploadFile = File(...)):
    """특성 추출 디버깅 전용 API"""
    temp_path = None
    try:
        # 파일 확장자 검사
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".wav", ".m4a", ".mp3"]:
            return JSONResponse(
                status_code=400,
                content={"error": "지원하지 않는 파일 형식입니다. .wav, .m4a, .mp3 파일만 업로드 가능합니다."}
            )

        # 임시 파일 저장
        temp_path = f"debug_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"디버깅용 파일 저장: {file.filename} -> {temp_path}")

        # 특성 추출
        feature_vector = extract_features(temp_path)
        if feature_vector is None:
            return JSONResponse(
                status_code=400,
                content={"error": "오디오 파일에서 특성 추출에 실패했습니다."}
            )

        # 특성 상세 정보 로깅
        log_feature_details(feature_vector)

        # 특성 저장
        json_filename = save_features_to_json(feature_vector, temp_path)

        # 결과 반환
        feature_array = feature_vector.to_array()
        feature_names = feature_vector.feature_names

        result = {
            "success": True,
            "filename": file.filename,
            "feature_count": len(feature_array),
            "features": {
                name: float(value) for name, value in zip(feature_names, feature_array)
            },
            "statistics": {
                "min": float(np.min(feature_array)),
                "max": float(np.max(feature_array)),
                "mean": float(np.mean(feature_array)),
                "std": float(np.std(feature_array)),
                "median": float(np.median(feature_array))
            },
            "quality_check": {
                "nan_count": int(np.sum(np.isnan(feature_array))),
                "inf_count": int(np.sum(np.isinf(feature_array))),
                "is_valid": bool(np.all(np.isfinite(feature_array)))
            }
        }

        if json_filename:
            result["saved_to"] = json_filename

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"디버깅 중 오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"디버깅 중 오류가 발생했습니다: {str(e)}"}
        )
    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")


# 예측 API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """오디오 파일을 분석하여 수박 당도를 예측"""
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "모델이 로드되지 않았습니다."}
        )

    temp_path = None
    try:
        # 파일 확장자 검사
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".wav", ".m4a", ".mp3"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "지원하지 않는 파일 형식입니다. .wav, .m4a, .mp3 파일만 업로드 가능합니다."
                }
            )

        # 임시 파일 저장
        temp_path = f"temp_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"파일 업로드 완료: {file.filename} -> {temp_path}")

        # 특성 추출
        feature_vector = extract_features(temp_path)
        if feature_vector is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "오디오 파일에서 특성 추출에 실패했습니다."}
            )

        feature_array = feature_vector.to_array().reshape(1, -1)

        # 디버깅: 모델이 기대하는 feature 개수, 실제 feature vector 정보 출력
        logger.info(f"[predict] model.n_features_in_: {getattr(model, 'n_features_in_', None)}")
        logger.info(f"[predict] feature_array.shape: {feature_array.shape}")
        logger.info(f"[predict] feature_array dtype: {feature_array.dtype}")

        # 예측
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0] if hasattr(model, 'predict_proba') else None

        # 예측 클래스 이름 (모델 클래스에서 가져오기)
        if hasattr(model, 'classes_'):
            predicted_class = int(model.classes_[prediction])
        else:
            predicted_class = f"class_{prediction}"

        logger.info(f"🎯 예측 결과: {prediction} (클래스: {predicted_class})")
        if probability is not None:
            logger.info(f"   - 확률 분포: {probability}")
            logger.info(f"   - 최대 확률: {max(probability):.4f}")

        # 결과 반환
        result = {
            "success": True,
            "filename": file.filename,
            "prediction": int(prediction),
            "predicted_class": predicted_class,
            "confidence": float(max(probability)) if probability is not None else None
        }

        # 확률 분포 정보 추가 (클래스별)
        if probability is not None and hasattr(model, 'classes_'):
            result["probabilities"] = {
                str(int(cls)): float(prob) for cls, prob in zip(model.classes_, probability)
            }

        # NumPy 인코더를 사용하여 JSON 직렬화
        return JSONResponse(content=json.loads(json.dumps(result, cls=NumpyEncoder)))

    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"예측 중 오류가 발생했습니다: {str(e)}"
            }
        )
    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")


if __name__ == "__main__":
    import socket


    def get_local_ip():
        """로컬 네트워크 IP 주소를 반환"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"


    # 실행 안내 정보 출력
    local_ip = get_local_ip()
    print(f"🍉 수박 당도 예측 서버 시작")
    print(f"   - 로컬: http://localhost:9001")
    print(f"   - 네트워크: http://{local_ip}:9001")
    print(f"   - 상태 확인: http://{local_ip}:9001/health")
    print(f"   - 특성 디버깅: http://{local_ip}:9001/debug-features")
    print(f"   - 예측 API: http://{local_ip}:9001/predict")

    uvicorn.run(app, host="0.0.0.0", port=9001)