#!/bin/bash
# ══════════════════════════════════════════════════════════════
# KEDA & KEDA HTTP Add-on 설치 스크립트 (Scale-to-Zero 지원)
# ══════════════════════════════════════════════════════════════

echo "==============================================================="
echo "1. Helm 저장소 추가 및 업데이트"
echo "==============================================================="
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

echo "==============================================================="
echo "2. KEDA (Kubernetes Event-driven Autoscaling) 코어 설치"
echo "==============================================================="
helm upgrade --install keda kedacore/keda --namespace keda --create-namespace

echo "==============================================================="
echo "3. KEDA HTTP Add-on 설치 (0-Scaling용 HTTP 인터셉터)"
echo "==============================================================="
helm upgrade --install http-add-on kedacore/keda-add-ons-http --namespace keda

echo "==============================================================="
echo "✅ 설치가 백그라운드에서 진행됩니다. 약 1~2분 뒤 완료됩니다!"
echo "명령어로 파드 상태를 확인하세요: kubectl get pods -n keda"
echo "==============================================================="
