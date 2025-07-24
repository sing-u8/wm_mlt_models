#!/usr/bin/env python3
"""
Simple test to verify feature extraction with one file
"""

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    exit(1)

try:
    import librosa
    print("✓ librosa imported successfully")
except ImportError as e:
    print(f"❌ librosa import failed: {e}")
    exit(1)

# Test loading and processing one audio file
test_file = "data/raw/train/watermelon_A/수박A_레오-01.wav"

try:
    print(f"Testing file: {test_file}")
    
    # Load audio with librosa
    y, sr = librosa.load(test_file, sr=22050, mono=True)
    print(f"✓ Audio loaded: shape={y.shape}, sr={sr}")
    
    # Test MFCC extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    print(f"✓ MFCC extracted: shape={mfcc_mean.shape}")
    
    # Test Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_spec_db)
    mel_std = np.std(mel_spec_db)
    print(f"✓ Mel Spectrogram: mean={mel_mean:.4f}, std={mel_std:.4f}")
    
    # Test Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(spec_centroid)
    print(f"✓ Spectral Centroid: {centroid_mean:.4f}")
    
    # Test Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    print(f"✓ Chroma extracted: shape={chroma_mean.shape}")
    
    # Create final feature vector
    feature_vector = np.concatenate([
        mfcc_mean,  # 13 features
        [mel_mean, mel_std, centroid_mean, 0.0, 0.0],  # 5 features (simplified)
        chroma_mean  # 12 features
    ])
    print(f"✓ Feature vector created: shape={feature_vector.shape}")
    print(f"✓ Expected 30 features, got {len(feature_vector)} features")
    
    if len(feature_vector) == 30:
        print("🎉 Feature extraction test PASSED!")
    else:
        print("❌ Feature vector size mismatch")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nSimple feature extraction test completed.")