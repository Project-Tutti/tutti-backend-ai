#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Tutti AI Server — 비용 최적화 GPU Node Pool 생성 스크립트 (Zero-Scaling)
# ══════════════════════════════════════════════════════════════

GCLOUD_CMD="/Users/eddy81848/Downloads/google-cloud-sdk/bin/gcloud"
if [ ! -f "$GCLOUD_CMD" ]; then
    GCLOUD_CMD="gcloud" # fallback
fi

PROJECT_ID=$($GCLOUD_CMD config get-value project)
CLUSTER_NAME="tutti-cluster"
ZONE="us-central1-a"
NODE_POOL_NAME="gpu-pool-t4" # T4 풀 명명

# 💡 모델 구조 대비 최적의 GPU 선택 (T4 vs L4)
# GPT 기반 미디 생성 모델(약 100M~300M 파라미터 내외)은 추론 연산량이 크지 않고,
# KV Cache를 접목하여 속도가 최적화되어 있으므로, L4(시간당 ~$0.60)보다
# 훨씬 저렴하고 안정적인 T4(시간당 ~$0.35) 1장으로도 충분히 목적 달성이 가능합니다.
# 또한 Spot 인스턴스로 요청 시 최대 70% 비용을 아낄 수 있습니다.

echo "==============================================================="
echo "1. Zero-Scaling GPU (T4) Node Pool 생성"
echo "==============================================================="
echo "특징:"
echo "- min-nodes: 0 (평소에는 노드가 없어 비용 $0 발생)"
echo "- max-nodes: 1 (요청이 쌓이면 최대 1대까지 자동 확장)"
echo "- machine-type: n1-standard-4 (T4 권장 조합)"
echo "- taints: 다른 잡다한 일반 파드들이 이 비싼 GPU 노드를 점유하지 않도록 방어 결계(Taint) 설정"

# gcloud 명령어로 GKE 클러스터에 새 노드 풀 추가
$GCLOUD_CMD container node-pools create $NODE_POOL_NAME \
    --project=$PROJECT_ID \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1,gpu-driver-version=DEFAULT \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=1 \
    --node-taints=nvidia.com/gpu=present:NoSchedule \
    --disk-size=50GB \
    --disk-type=pd-ssd

echo -e "\n완료되었습니다! 이제 deployment.yaml 에 NodeSelector와 Toleration이 올바르게 반영되어 있으면,"
echo "AI 편곡 요청이 들어와 Pod가 뜰 때 구글이 1~2분 내에 자동으로 GPU 노드와 드라이버를 구성해 줍니다."
