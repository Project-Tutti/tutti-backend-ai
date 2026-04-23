"""
worker.py — Redis Streams 기반 편곡 인퍼런스 워커

Redis Stream(ai:arrange:stream)에서 Consumer Group으로 작업을 소비하고,
편곡 인퍼런스를 수행한 뒤 콜백으로 결과를 전송합니다.

┌─────────────────────────────────────────────────────────────────────┐
│  신뢰성(Resilience) 보장 메커니즘                                      │
│                                                                     │
│  1. XREADGROUP: 메시지를 Consumer에 할당                               │
│     → 미ACK 시 자동으로 Pending 상태 유지 (메시지 증발 방지)                 │
│                                                                     │
│  2. XACK: 콜백 성공 후에만 호출                                         │
│     → 처리 도중 크래시 시 메시지는 Pending으로 잔존                         │
│                                                                     │
│  3. XPENDING + XCLAIM: 재시작 시 자동 복구                              │
│     → 10분 이상 Pending인 메시지를 현재 Worker가 인수                      │
│                                                                     │
│  4. 지수 백오프 재시도: 콜백 실패 시 5초→10초→20초...→최대5분                 │
│     → 총 30분 한도. 초과 시 XACK 처리 (백엔드 GC가 15분 후 FAILED)          │
│                                                                     │
│  5. 그레이스풀 셧다운: SIGTERM 시 현재 작업 완료 후 안전 종료                  │
└─────────────────────────────────────────────────────────────────────┘

사용법:
    python worker.py

환경변수:
    REDIS_HOST      — Redis 호스트 (default: localhost)
    REDIS_PORT      — Redis 포트 (default: 6379)
    REDIS_PASSWORD  — Redis 비밀번호
    REDIS_TLS       — TLS 사용 여부 (default: false)
"""

import os
import sys
import json
import time
import signal
import socket
import shutil
import uuid
import logging
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import redis
import requests

# ─── 프로젝트 내부 모듈 ───
from app.core.config import settings
from app.core.model_registry import ModelRegistry
from app.services.inference import run_arrangement
from app.services.midi_processor import remap_original_tracks
from app.schemas.request import Mapping


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════
STREAM_KEY = "ai:arrange:stream"
GROUP_NAME = "arrange-workers"
CONSUMER_NAME = f"{socket.gethostname()}-{os.getpid()}"

POLL_INTERVAL_SEC = 10                   # 새 메시지 폴링 간격 (Upstash 무료 500K cmd/월 고려)
PENDING_CLAIM_MIN_MS = 10 * 60 * 1000    # 10분: XCLAIM 임계값 (정상 인퍼런스가 뺏기지 않도록 여유 확보)

# 콜백 지수 백오프 (중요 콜백: 완료/실패)
BACKOFF_BASE_SEC = 5                     # 시작 대기: 5초
BACKOFF_MAX_SEC = 300                    # 최대 대기: 5분
RETRY_BUDGET_SEC = 1800                  # 총 재시도 한도: 30분

# 콜백 경량 재시도 (진행률 heartbeat)
HEARTBEAT_MAX_RETRIES = 3
HEARTBEAT_BACKOFF_SEC = 2


# ══════════════════════════════════════════════════════════════
# 로깅
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("worker")


# ══════════════════════════════════════════════════════════════
# HealthCheckServer — 컨테이너 헬스체크용 경량 HTTP 서버
# ══════════════════════════════════════════════════════════════
HEALTH_PORT = 8000

# 모듈 수준 상태 플래그 (메인 스레드에서 변경, 헬스체크 스레드에서 읽기)
_model_loaded = False


class _HealthHandler(BaseHTTPRequestHandler):
    """GET /health 에만 응답하는 최소 핸들러."""

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "worker_id": CONSUMER_NAME,
                "model_loaded": _model_loaded,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """헬스체크 로그 억제 (30초마다 찍히면 노이즈)."""
        pass


def _start_health_server():
    """데몬 스레드에서 헬스체크 HTTP 서버를 시작합니다."""
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"헬스체크 서버 시작: http://0.0.0.0:{HEALTH_PORT}/health")


# ══════════════════════════════════════════════════════════════
# CallbackClient — 동기 HTTP 콜백 전송
# ══════════════════════════════════════════════════════════════
class CallbackClient:
    """백엔드에 콜백을 전송하는 동기 HTTP 클라이언트.

    세 가지 전송 모드:
      - send_progress(): 진행률 heartbeat → 경량 재시도 (3회, 2초)
      - send_result():   완료 + MIDI 파일 → 지수 백오프 (30분 한도)
      - send_failure():  실패 알림        → 지수 백오프 (30분 한도)
    """

    def send_progress(self, url: str, secret: str, payload: dict) -> bool:
        """(A) 진행률 콜백 — 경량 재시도 (3회, 2초 간격).

        실패해도 치명적이지 않음 (백엔드 GC는 15분 기준).
        """
        for attempt in range(HEARTBEAT_MAX_RETRIES):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"X-Callback-Secret": secret},
                    timeout=10,
                )
                resp.raise_for_status()
                logger.info(
                    f"진행률 콜백 전송 성공: "
                    f"p{payload.get('projectId')}/v{payload.get('versionId')} "
                    f"→ {payload.get('progress')}%"
                )
                return True
            except Exception as e:
                if attempt < HEARTBEAT_MAX_RETRIES - 1:
                    logger.warning(
                        f"진행률 콜백 재시도 {attempt + 1}/{HEARTBEAT_MAX_RETRIES}: {e}"
                    )
                    time.sleep(HEARTBEAT_BACKOFF_SEC)
                else:
                    logger.warning(f"진행률 콜백 최종 실패 (비치명적): {e}")
        return False

    def send_result(
        self, url: str, secret: str, payload: dict, file_path: Path
    ) -> bool:
        """(B) 완료 콜백 — Multipart (MIDI + JSON). 지수 백오프 재시도."""
        if not file_path.exists():
            logger.error(f"전송할 파일이 존재하지 않습니다: {file_path}")
            return False

        def _do_send():
            with open(file_path, "rb") as f:
                resp = requests.post(
                    url,
                    data={"metadata": json.dumps(payload)},
                    files={"file": (file_path.name, f, "audio/midi")},
                    headers={"X-Callback-Secret": secret},
                    timeout=60,
                )
                resp.raise_for_status()
            logger.info(
                f"완료 콜백 전송 성공: "
                f"{file_path.name} ({file_path.stat().st_size:,} bytes)"
            )
            return True

        return self._retry_with_backoff(_do_send, "완료 콜백")

    def send_failure(self, url: str, secret: str, payload: dict) -> bool:
        """(A) 실패 콜백 — JSON POST. 지수 백오프 재시도."""

        def _do_send():
            resp = requests.post(
                url,
                json=payload,
                headers={"X-Callback-Secret": secret},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("실패 콜백 전송 성공")
            return True

        return self._retry_with_backoff(_do_send, "실패 콜백")

    def _retry_with_backoff(self, func, label: str) -> bool:
        """지수 백오프 재시도 엔진.

        대기 시간: 5초 → 10초 → 20초 → 40초 → 80초 → 160초 → 300초 → 300초 ...
        총 30분(1800초) 한도 초과 시 포기.
        """
        start = time.time()
        attempt = 0

        while True:
            try:
                return func()
            except Exception as e:
                elapsed = time.time() - start
                if elapsed >= RETRY_BUDGET_SEC:
                    break

                wait = min(BACKOFF_BASE_SEC * (2 ** attempt), BACKOFF_MAX_SEC)
                remaining = RETRY_BUDGET_SEC - elapsed
                actual_wait = min(wait, remaining)

                logger.warning(
                    f"[{label}] 재시도 #{attempt + 1}: {e}. "
                    f"{actual_wait:.0f}초 대기 "
                    f"(경과 {elapsed:.0f}초 / {RETRY_BUDGET_SEC}초)"
                )
                time.sleep(actual_wait)
                attempt += 1

        logger.error(
            f"[{label}] ═══════════════════════════════════════\n"
            f"  콜백 최종 실패 — 30분간 재시도 소진\n"
            f"  백엔드 GC가 15분 후 자동 FAILED 처리합니다.\n"
            f"═══════════════════════════════════════"
        )
        return False


# ══════════════════════════════════════════════════════════════
# MIDI 다운로드 (동기)
# ══════════════════════════════════════════════════════════════
def _download_midi_sync(url: str) -> Path:
    """MIDI 파일을 동기적으로 다운로드합니다.

    기존 midi_processor.py의 async 버전 대신
    Worker의 동기 루프에 맞는 requests 기반 구현.
    """
    temp_dir = Path("/tmp/tutti_midi_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    fd, path_str = tempfile.mkstemp(suffix=".mid", dir=temp_dir)
    os.close(fd)
    file_path = Path(path_str)

    logger.info(f"MIDI 다운로드: {url}")
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(
            f"MIDI 다운로드 완료: {file_path} ({file_path.stat().st_size:,} bytes)"
        )
        return file_path
    except Exception:
        if file_path.exists():
            file_path.unlink()
        raise


# ══════════════════════════════════════════════════════════════
# 작업 처리 (process_job)
# ══════════════════════════════════════════════════════════════
def process_job(
    job_data: dict,
    registry: ModelRegistry,
    callback: CallbackClient,
) -> bool:
    """단일 편곡 작업 처리.

    arrangement.py의 process_arrangement()를 동기 버전으로 재구성.
    async 환경의 복잡한 코루틴 브릿지 없이 직접 동기 호출.

    Returns:
        항상 True. 성공이든 실패든 "처리 완료"로 간주 → XACK 허용.
        실패한 작업은 콜백으로 알리고, 백엔드 GC가 최종 안전장치.
    """
    job_id = str(uuid.uuid4())[:8]
    cb_url = job_data["callbackUrl"]
    cb_secret = job_data["callbackSecret"]
    project_id = job_data["projectId"]
    version_id = job_data["versionId"]

    midi_path = None
    mapped_midi_path = None
    result_path = None

    def _payload(status: str, progress: int, **extra) -> dict:
        p = {
            "projectId": project_id,
            "versionId": version_id,
            "status": status,
            "progress": progress,
        }
        p.update(extra)
        return p

    try:
        # ── Step 1: MIDI 다운로드 (10%) ──────────────────────
        logger.info(f"[{job_id}] Step 1/4: MIDI 다운로드")
        midi_path = _download_midi_sync(job_data["midiFilePath"])

        callback.send_progress(cb_url, cb_secret, _payload("processing", 10))

        # ── Step 2: 트랙 재매핑 검증 및 전처리 (20%) ────────────────
        logger.info(f"[{job_id}] Step 2/4: 트랙 재매핑 검증")
        mappings = [Mapping(**m) for m in job_data.get("mappings", [])]
        
        from app.services.midi_processor import is_mapping_noop
        import mido
        tmp_mid = mido.MidiFile(str(midi_path))
        
        if is_mapping_noop(tmp_mid, mappings):
            logger.info(f"[{job_id}] ↳ 맵핑 데이터가 원본과 동일합니다(NO-OP). 전처리를 생략하고 원본을 사용합니다.")
            inference_path = midi_path
            mapped_midi_path = None
        else:
            logger.info(f"[{job_id}] ↳ 커스텀 맵핑 감지. {len(mappings)}개 트랙 재매핑 수행")
            mapped_midi_path = midi_path.with_name(midi_path.stem + "_mapped.mid")
            shutil.copy2(midi_path, mapped_midi_path)
            remap_original_tracks(mapped_midi_path, mappings)
            inference_path = mapped_midi_path

        callback.send_progress(cb_url, cb_secret, _payload("processing", 20))

        # ── Step 3: 추론 (20% → 80%) ────────────────────────
        target_prog = int(job_data["targetInstrumentId"])
        loaded = registry.get_model(job_data.get("modelType"))

        logger.info(
            f"[{job_id}] Step 3/4: 추론 시작 "
            f"(target_prog={target_prog}, model={loaded.name}, "
            f"genre={job_data.get('genre', 'CLASSICAL')}, "
            f"temp={job_data.get('temperature', 1.0)})"
        )

        output_dir = Path(settings.RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{job_id}.mid"

        # 동기 progress_hook — 인퍼런스 슬라이딩 윈도우마다 heartbeat 전송
        def progress_hook(pct: int):
            callback.send_progress(
                cb_url, cb_secret, _payload("processing", pct)
            )

        target_midi_program_raw = job_data.get("targetMidiProgram")
        target_midi_program = None
        if target_midi_program_raw is not None:
            try:
                target_midi_program = int(target_midi_program_raw)
            except ValueError:
                logger.warning(f"[{job_id}] 유효하지 않은 targetMidiProgram 무시: {target_midi_program_raw}")

        result_file = run_arrangement(
            song_path=str(inference_path),
            target_prog=target_prog,
            genre=job_data.get("genre", "CLASSICAL"),
            temperature=job_data.get("temperature", 1.0),
            pitch_min=job_data.get("minNote"),
            pitch_max=job_data.get("maxNote"),
            output_path=str(result_path),
            model=loaded.model,
            vocab=loaded.vocab,
            vocab_r=loaded.vocab_r,
            device=loaded.device,
            progress_hook=progress_hook,
            original_song_path=str(midi_path),
            actual_instrument_name=job_data.get("targetInstrumentName"),
            actual_midi_program=target_midi_program,
        )

        callback.send_progress(cb_url, cb_secret, _payload("processing", 80))

        # ── Step 4: 완료 콜백 (100%) ─────────────────────────
        logger.info(f"[{job_id}] Step 4/4: 결과 MIDI 전송")
        success = callback.send_result(
            cb_url,
            cb_secret,
            _payload("complete", 100),
            file_path=Path(result_file),
        )

        if success:
            logger.info(f"[{job_id}] ✅ 편곡 완료 — 콜백 전송 성공")
        else:
            logger.error(
                f"[{job_id}] ⚠️ 편곡은 완료했으나 콜백 전송 실패 — "
                f"백엔드 GC가 처리합니다"
            )

    except Exception as e:
        logger.error(f"[{job_id}] ❌ 편곡 실패: {e}", exc_info=True)
        callback.send_failure(
            cb_url,
            cb_secret,
            _payload("failed", 0, errorMessage=str(e)),
        )

    finally:
        # 임시 파일 정리
        for path in [result_path, midi_path, mapped_midi_path]:
            try:
                if path and Path(path).exists():
                    Path(path).unlink()
                    logger.debug(f"[{job_id}] 임시 파일 삭제: {path}")
            except Exception:
                pass

    return True


# ══════════════════════════════════════════════════════════════
# RedisStreamConsumer — Redis Streams 메시지 소비자
# ══════════════════════════════════════════════════════════════
class RedisStreamConsumer:
    """Redis Streams Consumer Group 기반 메시지 소비자.

    Upstash 서버리스 제약:
      - BLOCK 옵션 사용 불가 → 논블로킹 + time.sleep() 폴링
      - XREADGROUP, XACK, XPENDING, XCLAIM 모두 정상 지원
    """

    def __init__(self):
        self._redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", ""),
            ssl=os.getenv("REDIS_TLS", "false").lower() == "true",
            decode_responses=True,
        )
        self._consumer = CONSUMER_NAME
        self._ensure_group()

    def _ensure_group(self):
        """Consumer Group 생성 (이미 존재하면 무시)."""
        try:
            self._redis.xgroup_create(
                STREAM_KEY, GROUP_NAME, id="0", mkstream=True
            )
            logger.info(f"Consumer Group 생성: {GROUP_NAME} (Stream: {STREAM_KEY})")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer Group 이미 존재: {GROUP_NAME}")
            else:
                raise

    def recover_pending(self) -> list:
        """이전 Consumer의 미처리(Pending) 메시지를 XCLAIM으로 복구.

        10분 이상 idle 상태인 메시지만 인수합니다.
        다른 Worker가 정상 처리 중인 메시지를 강탈하지 않기 위함.
        """
        try:
            summary = self._redis.xpending(STREAM_KEY, GROUP_NAME)
            pending_count = summary.get("pending", 0)

            if pending_count == 0:
                logger.info("복구할 Pending 메시지 없음")
                return []

            logger.info(f"Pending 메시지 {pending_count}개 발견, 복구 검토 중...")

            # 상세 Pending 목록 조회 (최대 100개)
            details = self._redis.xpending_range(
                STREAM_KEY, GROUP_NAME, min="-", max="+", count=100
            )

            recovered = []
            for entry in details:
                idle_ms = entry.get("time_since_delivered", 0)
                if idle_ms < PENDING_CLAIM_MIN_MS:
                    logger.debug(
                        f"  Pending {entry['message_id']}: "
                        f"idle {idle_ms // 1000}초 < 임계값 {PENDING_CLAIM_MIN_MS // 1000}초 → 건너뜀"
                    )
                    continue

                msg_id = entry["message_id"]
                original_consumer = entry.get("consumer", "unknown")

                # XCLAIM: 이 Consumer가 소유권 인수
                claimed = self._redis.xclaim(
                    STREAM_KEY,
                    GROUP_NAME,
                    self._consumer,
                    min_idle_time=PENDING_CLAIM_MIN_MS,
                    message_ids=[msg_id],
                )

                if claimed:
                    recovered.extend(claimed)
                    logger.info(
                        f"  ↳ Pending 메시지 복구: {msg_id} "
                        f"(idle {idle_ms // 1000}초, "
                        f"원래 consumer: {original_consumer})"
                    )

            if recovered:
                logger.info(f"총 {len(recovered)}개 메시지 복구 완료")
            else:
                logger.info(
                    "Pending 메시지가 있지만 모두 10분 미만 — 다른 Worker가 처리 중"
                )

            return recovered

        except Exception as e:
            logger.error(f"Pending 메시지 복구 실패: {e}", exc_info=True)
            return []

    def fetch_job(self):
        """새 메시지 1개 가져오기 (논블로킹).

        Returns:
            (msg_id, fields_dict) 또는 None
        """
        try:
            # BLOCK 옵션 없이 호출 → Upstash 호환
            result = self._redis.xreadgroup(
                GROUP_NAME,
                self._consumer,
                {STREAM_KEY: ">"},
                count=1,
            )

            if not result:
                return None

            # result = [(stream_name, [(msg_id, {field: value}), ...])]
            _, messages = result[0]
            if not messages:
                return None

            msg_id, fields = messages[0]
            return msg_id, fields

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis 연결 끊김: {e}")
            return None
        except Exception as e:
            logger.error(f"메시지 가져오기 실패: {e}", exc_info=True)
            return None

    def ack(self, msg_id: str):
        """메시지 처리 완료 확인 (XACK).

        이 호출 후 메시지는 Consumer Group의 Pending 목록에서 제거됩니다.
        Stream 자체에는 남아 있어 디버깅 시 XRANGE로 확인 가능.
        """
        try:
            self._redis.xack(STREAM_KEY, GROUP_NAME, msg_id)
            logger.info(f"XACK 완료: {msg_id}")
        except Exception as e:
            logger.error(f"XACK 실패: {msg_id} — {e}")

    def ping(self) -> bool:
        """Redis 연결 확인."""
        try:
            return self._redis.ping()
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════
# 메시지 파싱 & 처리 유틸리티
# ══════════════════════════════════════════════════════════════
def _parse_stream_message(fields: dict) -> dict:
    """Stream entry의 fields에서 작업 JSON을 추출합니다.

    백엔드가 XADD 시 사용하는 구조:
        XADD ai:arrange:stream * data <JSON_STRING>
    → fields = {"data": "<JSON_STRING>"}
    """
    raw = fields.get("data")
    if not raw:
        raise ValueError(f"메시지에 'data' 필드가 없습니다: {fields}")
    return json.loads(raw)


def _handle_message(
    msg_id: str,
    fields: dict,
    registry: ModelRegistry,
    callback: CallbackClient,
    consumer: RedisStreamConsumer,
):
    """메시지 파싱 → process_job → XACK."""
    try:
        job_data = _parse_stream_message(fields)
        logger.info(
            f"작업 수신: msg_id={msg_id}, "
            f"project={job_data.get('projectId')}, "
            f"version={job_data.get('versionId')}, "
            f"target={job_data.get('targetInstrumentId')}"
        )
    except Exception as e:
        logger.error(
            f"메시지 파싱 실패: msg_id={msg_id}, error={e}. "
            f"복구 불가능 — XACK 처리"
        )
        consumer.ack(msg_id)
        return

    # 작업 처리 (성공/실패 모두 True 반환)
    process_job(job_data, registry, callback)

    # 콜백까지 완료된 후에만 ACK
    consumer.ack(msg_id)


# ══════════════════════════════════════════════════════════════
# 메인 진입점
# ══════════════════════════════════════════════════════════════
def main():
    logger.info("═" * 60)
    logger.info("  Tutti AI Worker 시작 (Redis Streams)")
    logger.info(f"  Consumer : {CONSUMER_NAME}")
    logger.info(f"  Stream   : {STREAM_KEY}")
    logger.info(f"  Group    : {GROUP_NAME}")
    logger.info(f"  Poll     : {POLL_INTERVAL_SEC}초 간격")
    logger.info(f"  Redis    : {os.getenv('REDIS_HOST', 'localhost')}:"
                f"{os.getenv('REDIS_PORT', '6379')} "
                f"(TLS: {os.getenv('REDIS_TLS', 'false')})")
    logger.info("═" * 60)

    # ── 그레이스풀 셧다운 시그널 핸들러 ──
    running = True
    processing = False

    def _shutdown(signum, frame):
        nonlocal running
        sig_name = signal.Signals(signum).name
        if processing:
            logger.info(
                f"📛 {sig_name} 수신 — 현재 작업이 끝나면 종료합니다..."
            )
        else:
            logger.info(f"📛 {sig_name} 수신 — 즉시 종료합니다")
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # ── 0단계: 헬스체크 서버 시작 (모델 로드 전) ──
    _start_health_server()

    # ── 1단계: 모델 로드 ──
    global _model_loaded
    logger.info("🔧 모델 로딩 시작...")
    model_dir = Path(settings.MODEL_DIR)
    registry = ModelRegistry(model_dir)
    registry.load_all_models()
    _model_loaded = True
    logger.info("🔧 모델 로딩 완료 → 헬스체크 model_loaded=true")

    # ── 2단계: Redis 연결 ──
    consumer = RedisStreamConsumer()
    callback_client = CallbackClient()

    if consumer.ping():
        logger.info("🟢 Redis 연결 확인")
    else:
        logger.error("🔴 Redis 연결 실패! Worker를 종료합니다.")
        sys.exit(1)

    # ── 3단계: 크래시 복구 (Pending 메시지 재처리) ──
    logger.info("🔄 크래시 복구 점검...")
    recovered = consumer.recover_pending()
    for msg_id, fields in recovered:
        if not running:
            break
        processing = True
        logger.info(f"🔄 복구 메시지 처리 중: {msg_id}")
        _handle_message(msg_id, fields, registry, callback_client, consumer)
        processing = False

    # ── 4단계: 메인 루프 ──
    logger.info("🎵 메인 루프 시작 — 새 작업을 기다리는 중...")
    last_pending_check = time.time()

    while running:
        # 주기적 크래시 복구 점검 (1분마다 실행)
        if time.time() - last_pending_check > 60:
            recovered = consumer.recover_pending()
            for msg_id, fields in recovered:
                if not running:
                    break
                processing = True
                logger.info(f"🔄 복구 메시지 처리 중: {msg_id}")
                _handle_message(msg_id, fields, registry, callback_client, consumer)
                processing = False
            last_pending_check = time.time()

        result = consumer.fetch_job()

        if result is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        msg_id, fields = result
        processing = True
        _handle_message(msg_id, fields, registry, callback_client, consumer)
        processing = False

    logger.info("Worker 정상 종료 👋")


if __name__ == "__main__":
    main()
