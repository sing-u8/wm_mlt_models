#!/bin/bash

# SSL 인증서 생성 스크립트
# 자체 서명 인증서를 생성합니다

echo "🔐 자체 서명 SSL 인증서 생성 중..."

# 인증서를 저장할 디렉토리 생성
mkdir -p ssl

# 개인 키와 인증서 생성
openssl req -x509 -newkey rsa:4096 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -days 365 \
  -nodes \
  -subj "/C=KR/ST=Seoul/L=Seoul/O=WatermelonMLT/OU=Development/CN=localhost"

echo "✅ SSL 인증서 생성 완료!"
echo "   - 개인 키: ssl/key.pem"
echo "   - 인증서: ssl/cert.pem"
echo "   - 유효 기간: 365일"

# 인증서 정보 확인
echo ""
echo "📋 인증서 정보:"
openssl x509 -in ssl/cert.pem -text -noout | grep -E "(Subject:|Not Before:|Not After:)"