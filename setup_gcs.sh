#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Tutti AI Server — GCS 모델 버킷 생성 및 IAM 설정 스크립트 (비용 최적화)
# ══════════════════════════════════════════════════════════════

# 설정 변수
PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="tutti-ai-models"
LOCATION="us-central1" # GKE 클러스터와 동일한 리전을 사용해야 외부 네트워크 송신(Egress) 비용이 무료입니다!
GCP_SA_NAME="github-actions" # Github Actions에서 쓰고 있는 SA 이름으로 변경하세요. (예: github-actions-sa)
K8S_NAMESPACE="tutti"
K8S_SA_NAME="ai-server-sa"

echo "==============================================================="
echo "1. 비용 최적화 GCS 버킷 생성"
echo "==============================================================="
echo "💡 비용 최적화 팁:"
echo "   - Single-region (us-central1) 사용으로 스토리지 단가를 낮춥니다."
echo "   - GKE가 같은 us-central1 에 있으면 네트워크 송신 데이터 통신료가 $0 입니다."
echo "   - Standard Storage 클래스를 사용하여 잦은 읽기 시 검색 비용을 최소화합니다."

# 버킷 생성 (단일 리전, Standard Storage 클래스, 공개 접근 차단)
gcloud storage buckets create gs://$BUCKET_NAME \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access

# (선택) 만약 모델 파일이 자주 바뀌지 않는다면 버전 관리는 끄는 것이 용량 절감에 유리합니다.
gcloud storage buckets update gs://$BUCKET_NAME --no-versioning


echo -e "\n==============================================================="
echo "2. Workload Identity 연동 (Init Container 다운로드 권한)"
echo "==============================================================="

# K8s Service Account 생성 (중복 방지를 위해 이미 있으면 경고만 무시)
kubectl create serviceaccount $K8S_SA_NAME -n $K8S_NAMESPACE 2>/dev/null || true

# K8s SA <-> Google IAM SA 바인딩
# 주의: 워크로드 아이덴티티를 위해 보통 전용 Google IAM SA를 하나 만듭니다.
# 여기서는 편의상 프로젝트에 이미 있는 SA를 사용하거나 따로 만들어야 합니다.
GSA_EMAIL="${GCP_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# IAM SA에 버킷 객체 관리자 권한 부여 (GitHub Actions에서 업로드도 해야 하므로 Admin 부여)
gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
    --member="serviceAccount:$GSA_EMAIL" \
    --role="roles/storage.objectAdmin"

# IAM SA와 Kubernetes SA 연동 (Workload Identity)
gcloud iam service-accounts add-iam-policy-binding $GSA_EMAIL \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[$K8S_NAMESPACE/$K8S_SA_NAME]"

# K8s SA에 어노테이션 추가
kubectl annotate serviceaccount $K8S_SA_NAME -n $K8S_NAMESPACE \
    iam.gke.io/gcp-service-account=$GSA_EMAIL --overwrite


echo -e "\n==============================================================="
echo "3. 모델 업로드 방법 (가이드)"
echo "==============================================================="
echo "이제 다음 명령어를 통해 로컬의 모델들을 업로드하세요."
echo ""
echo "$ gsutil cp registry.json gs://$BUCKET_NAME/v1/registry.json"
echo "$ gsutil cp piano_v1.pt gs://$BUCKET_NAME/v1/piano_v1.pt"
echo ""
echo "완료되었습니다!"
