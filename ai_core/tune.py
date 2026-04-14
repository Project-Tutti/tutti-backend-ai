"""ai_core/tune.py — Optuna 하이퍼파라미터 자동 튜닝 + MLflow 기록.

GPU 서버에서 실행하는 스크립트.
Optuna가 편곡 파라미터(temperature, top_p, rest_penalty 등)를 자동 탐색하고,
모든 트라이얼을 MLflow에 기록합니다.

스터디는 SQLite에 영속 저장되어 optuna-dashboard로 시각화할 수 있습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 핵심 최적화 기능 3가지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) 조기 종료 (Pruning) — --pruning
   → 파일 단위로 중간 결과를 보고하다가 형편없으면 즉시 중단 → 다음 트라이얼로.
   → Optuna MedianPruner 사용: 지금까지의 중간값이 전체 트라이얼 중간값보다 나쁘면 프루닝.
   ┌──────────────────────────────────────────┐
   │ 장점: 명백히 나쁜 조합에 시간 낭비 안 함      │
   │ 단점: 초반 파일 결과에 치우칠 수 있음          │
   │ 권장: 파일 5개 이상일 때 효과적               │
   └──────────────────────────────────────────┘

2) 데이터 샘플링 (Subset) — --sample-ratio N
   → 전체 데이터 중 N% 만 랜덤 샘플링해서 빠르게 경향성 파악.
   → 트라이얼마다 다른 서브셋을 뽑아 과적합 방지.
   ┌──────────────────────────────────────────┐
   │ 장점: 데이터 많을 때 극적 속도 향상           │
   │       (100곡 → 10곡만 = 10배 빠름)           │
   │ 단점: 샘플이 편향되면 최적값 정확도↓          │
   │ 권장: 탐색 → sample-ratio 0.1~0.3           │
   │       검증 → sample-ratio 1.0 (전체)        │
   └──────────────────────────────────────────┘

3) 파일 수 제한 (Max Files) — --max-files N
   → 트라이얼당 최대 N개 파일만 평가.
   → 전체 데이터가 많을 때 정식 탐색 전 빠르게 5~10곡으로 테스트.
   ┌──────────────────────────────────────────┐
   │ 장점: 예측 가능한 실행 시간                   │
   │       (N곡 × 30초 × 트라이얼 수 = 총 시간)   │
   │ 단점: 평가 곡 수가 적으면 일반화 약함          │
   │ 권장: 빠른 탐색 5곡 → 정밀 검증 전체           │
   └──────────────────────────────────────────┘

사용법:
    # 기본 (30 트라이얼, 전체 파일)
    python -m ai_core.tune \\
        --source ~/midi_data/test_songs/ \\
        --target keyboard --n-trials 30

    # ⚡ 빠른 탐색: 10% 샘플 + 프루닝 + 최대 5곡
    python -m ai_core.tune \\
        --source ~/midi_data/test_songs/ \\
        --target keyboard --n-trials 50 \\
        --pruning --sample-ratio 0.1 --max-files 5

    # 🔍 정밀 검증: 전체 데이터로 30 트라이얼 (프루닝만)
    python -m ai_core.tune \\
        --source ~/midi_data/test_songs/ \\
        --target keyboard --n-trials 30 \\
        --pruning

    # 대시보드 (별도 터미널)
    optuna-dashboard sqlite:///optuna_studies.db

의존성:
    uv sync --extra gpu --extra mlflow
"""

import argparse
import gc
import logging
import math
import os
import random as _random
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# 기본 SQLite DB 경로
DEFAULT_DB_PATH = "optuna_studies.db"

# ──────────────────────────────────────────────
# 튜닝 가능 파라미터 정의
# ──────────────────────────────────────────────
TUNABLE_PARAMS = {
    "temperature": {"type": "float", "low": 0.5, "high": 1.5, "step": 0.05},
    "top_p":       {"type": "float", "low": 0.8, "high": 0.99, "step": 0.01},
    "rest_penalty": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.25},
    "fade_bars":   {"type": "int",   "low": 4,   "high": 16,  "step": 4},
    "window_bars": {"type": "int",   "low": 4,   "high": 16,  "step": 4},
    "context_bars": {"type": "int",   "low": 4,   "high": 16,  "step": 4},
}

# 최적화할 수 있는 메트릭
METRIC_DIRECTIONS = {
    "chord_accuracy":  "maximize",
    "pch_similarity":  "maximize",
    "doa":             "maximize",
    "dissonance_rate": "minimize",
}


def _load_model(model_type: str | None = None):
    """모델을 로드합니다. (최초 1회만)"""
    from app.core.config import settings
    from app.core.model_registry import ModelRegistry

    model_dir = Path(settings.MODEL_DIR)
    registry = ModelRegistry(model_dir)
    registry.load_all_models()
    loaded = registry.get_model(model_type)
    logger.info(f"모델 로드 완료: {loaded.name} (device={loaded.device})")
    return loaded


def _collect_source_files(source_path: str) -> list[Path]:
    """소스 MIDI 파일 목록을 수집합니다."""
    p = Path(source_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(p.glob("*.mid")) + sorted(p.glob("*.midi"))
        if not files:
            logger.error(f"디렉토리에 MIDI 파일이 없습니다: {p}")
            sys.exit(1)
        return files
    else:
        logger.error(f"경로를 찾을 수 없습니다: {p}")
        sys.exit(1)


def _sample_files(
    source_files: list[Path],
    sample_ratio: float,
    max_files: int | None,
) -> list[Path]:
    """
    데이터 샘플링 + 파일 수 제한을 적용합니다.

    ── 처리 순서 ──
    1. sample_ratio < 1.0 이면 → 전체의 N%를 랜덤 추출 (매번 다른 셔플)
    2. max_files가 있으면 → 추출 결과를 max_files로 잘라냄
    두 옵션은 독립적으로, 또는 함께 쓸 수 있음.

    Args:
        source_files: 전체 소스 MIDI 파일 리스트
        sample_ratio: 0.0~1.0 사이 비율 (1.0이면 전체 사용)
        max_files: 최대 파일 수 (None이면 제한 없음)

    Returns:
        선택된 파일 리스트 (원본은 변경하지 않음)
    """
    files = list(source_files)  # 원본 보존 위해 복사

    # ── 1) 데이터 샘플링 ──
    # 트라이얼마다 다른 서브셋을 뽑기 위해 매번 셔플.
    # 같은 서브셋만 쓰면 특정 곡에 과적합될 위험.
    if sample_ratio < 1.0:
        n_sample = max(1, int(len(files) * sample_ratio))
        _random.shuffle(files)
        files = files[:n_sample]

    # ── 2) 파일 수 제한 ──
    # sample_ratio와 독립적으로 하드 리밋을 걸어
    # 예측 가능한 실행 시간 보장 (N곡 × 추론시간 × 트라이얼수)
    if max_files is not None and len(files) > max_files:
        _random.shuffle(files)
        files = files[:max_files]

    return files


def _run_trial_inference(source_file: Path, target: str, genre: str,
                         loaded, params: dict,
                         output_dir: Path,
                         trial_number: int = 0) -> dict:
    """단일 트라이얼의 추론을 실행하고 메트릭을 반환합니다."""
    from ai_core.metrics import compute_basic_quality_metrics, compute_musical_quality

    output_path = output_dir / f"{source_file.stem}_tuned.mid"

    try:
        import random
        import torch
        from ai_core.tokenizer import midi_to_bar_tokens
        from ai_core.generator import generate_for_target
        from ai_core.postprocess import postprocess
        from ai_core.midi_writer import save_midi
        from ai_core.constants import INSTRUMENT_GROUPS

        cfg = INSTRUMENT_GROUPS[target]
        target_prog = cfg["representative"]
        pitch_min = cfg["pitch_min"]
        pitch_max = cfg["pitch_max"]
        _device = loaded.device

        # 트라이얼별 다른 시드 → 같은 파라미터라도 다른 샘플링 결과
        seed = 42 + trial_number
        random.seed(seed)
        torch.manual_seed(seed)

        header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(
            str(source_file), genre, loaded.vocab
        )

        all_notes = generate_for_target(
            loaded.model, header, bar_tokens, max_bar,
            target_prog, pitch_min, pitch_max,
            params["window_bars"], params["context_bars"],
            params["temperature"], params["top_p"],
            loaded.vocab, loaded.vocab_r, source_pm, _device,
            rest_penalty=params["rest_penalty"],
            fade_bars=params["fade_bars"],
        )

        all_notes = postprocess(all_notes, pitch_min, pitch_max, target_name=target)

        if len(all_notes) == 0:
            logger.warning(f"빈 결과 — {source_file.name}")
            return {}

        save_midi(
            all_notes, source_pm, str(output_path),
            target_prog, target,
            original_song_path=str(source_file),
        )

        # 메트릭 계산 — save_midi()는 반환값이 없으므로 output_path 직접 사용
        basic = compute_basic_quality_metrics(str(output_path))
        musical = compute_musical_quality(
            source_path=str(source_file),
            generated_path=str(output_path),
        )
        basic.update(musical)
        return basic

    except Exception as e:
        logger.error(f"트라이얼 추론 실패: {e}", exc_info=True)
        return {}
    finally:
        # GPU VRAM 해제 — KV cache 등 GPU 텐서가 즉시 해제되도록 강제
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def create_objective(
    source_files: list[Path],
    target: str,
    genre: str,
    loaded,
    optimize_metric: str,
    output_dir: Path,
    *,
    enable_pruning: bool = False,
    sample_ratio: float = 1.0,
    max_files: int | None = None,
):
    """Optuna objective 함수를 생성합니다.

    ── 최적화 전략 옵션 ──

    enable_pruning (조기 종료):
        True이면 파일 하나를 평가할 때마다 중간 결과를 Optuna에 보고합니다.
        지금까지의 누적 점수가 다른 트라이얼의 중간값보다 나쁘면(MedianPruner),
        남은 파일을 건너뛰고 즉시 다음 트라이얼로 넘어갑니다.

        ┌─────────────────────────────────────────────────────┐
        │ 동작 원리 (MedianPruner):                             │
        │                                                     │
        │  Trial A:  곡1=0.7  곡2=0.6  곡3=0.5  → 평균 0.60   │
        │  Trial B:  곡1=0.3  → 중간값(0.7)보다 나쁨 → ✂ 프루닝 │
        │  Trial C:  곡1=0.8  곡2=0.7  → 계속...               │
        │                                                     │
        │ n_startup_trials=3: 처음 3개는 프루닝하지 않고 기준 축적 │
        │ n_warmup_steps=1:   최소 1곡은 평가한 후 프루닝 판단    │
        └─────────────────────────────────────────────────────┘

        ⚠️ 주의: 프루닝된 트라이얼은 '불완전' 상태로 기록됩니다.
                 최종 best_trial에는 포함되지 않습니다.

    sample_ratio (데이터 샘플링):
        0.0~1.0 사이 비율. 예: 0.1이면 전체의 10%만 사용.
        트라이얼마다 다른 서브셋을 뽑아서 과적합을 방지합니다.

        ┌──────────────────────────────────────────────┐
        │ 추천 워크플로우:                                │
        │  Phase 1: --sample-ratio 0.1 --n-trials 100  │
        │           → 넓은 탐색, 유망한 영역 발견         │
        │  Phase 2: --sample-ratio 1.0 --n-trials 20   │
        │           → 좁은 범위에서 전체 데이터로 정밀 검증 │
        └──────────────────────────────────────────────┘

    max_files (파일 수 제한):
        트라이얼당 최대 평가 파일 수. sample_ratio와 독립적으로 적용.
        실행 시간을 예측 가능하게 만듭니다.

        예: 100곡 데이터 + --max-files 5 + --n-trials 30
           = 5곡 × ~30초 × 30회 ≈ 75분 (전체 사용 시 ~25시간)
    """

    def objective(trial):
        # ── 하이퍼파라미터 샘플링 ──
        params = {}
        for name, config in TUNABLE_PARAMS.items():
            if config["type"] == "float":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"], step=config["step"]
                )
            else:
                params[name] = trial.suggest_int(
                    name, config["low"], config["high"], step=config["step"]
                )

        logger.info(
            f"[Trial {trial.number}] "
            f"temp={params['temperature']:.2f}, "
            f"top_p={params['top_p']:.2f}, "
            f"rest_penalty={params['rest_penalty']:.2f}"
        )

        # ── 데이터 샘플링 + 파일 수 제한 적용 ──
        # 매 트라이얼마다 새로운 서브셋을 랜덤 추출합니다.
        # 같은 서브셋만 반복하면 특정 곡에 과적합되므로,
        # 트라이얼마다 다른 곡을 볼 수 있게 셔플합니다.
        trial_files = _sample_files(source_files, sample_ratio, max_files)
        if len(trial_files) < len(source_files):
            logger.info(
                f"[Trial {trial.number}] "
                f"📊 샘플링: {len(trial_files)}/{len(source_files)}곡 선택"
            )

        # ── 추론 + 평가 (프루닝 지원) ──
        all_scores = []
        for step, source_file in enumerate(trial_files):
            metrics = _run_trial_inference(
                source_file, target, genre, loaded, params, output_dir,
                trial_number=trial.number,
            )

            if metrics and optimize_metric in metrics:
                score = metrics[optimize_metric]
                all_scores.append(score)

                # MLflow 개별 파일 결과: step 기반 기록 (nested run 과다 생성 방지)
                _log_file_metric_to_mlflow(
                    trial, step, source_file.name, optimize_metric, score,
                )

                # ── 조기 종료 (Pruning) ──
                # 파일 하나를 평가할 때마다 중간 결과를 Optuna에 보고합니다.
                # Optuna는 이 중간값을 다른 완료된 트라이얼과 비교하여,
                # 현재 조합이 유망하지 않으면 TrialPruned를 발생시킵니다.
                #
                # 예시: 10곡을 평가하는데, 3곡째까지 chord_accuracy가 0.2면
                #       이미 완료된 다른 트라이얼들의 3곡 시점 중간값(0.5)보다 나쁨
                #       → 나머지 7곡 건너뛰고 즉시 다음 파라미터 조합으로 이동
                if enable_pruning and len(trial_files) > 1:
                    # step 기반 중간 값 보고 (누적 평균)
                    running_avg = sum(all_scores) / len(all_scores)
                    trial.report(running_avg, step)

                    # Optuna가 "이 트라이얼은 가망 없다"고 판단하면 프루닝
                    if trial.should_prune():
                        logger.info(
                            f"[Trial {trial.number}] ✂ 프루닝됨 "
                            f"(step {step + 1}/{len(trial_files)}, "
                            f"현재 {optimize_metric}={running_avg:.4f})"
                        )
                        # TrialPruned를 raise하면 Optuna가 이 트라이얼을 PRUNED로 기록
                        import optuna
                        raise optuna.TrialPruned()

        if not all_scores:
            logger.warning(f"[Trial {trial.number}] 유효한 결과 없음")
            import optuna
            raise optuna.TrialPruned("유효한 결과 없음")

        avg_score = sum(all_scores) / len(all_scores)
        logger.info(
            f"[Trial {trial.number}] 결과: "
            f"{optimize_metric}={avg_score:.4f} ({len(all_scores)}곡 평균)"
        )
        return avg_score

    return objective


def _log_file_metric_to_mlflow(trial, step: int, source_name: str,
                                metric_name: str, score: float):
    """개별 파일의 메트릭을 step 기반으로 기록합니다.

    nested run을 파일마다 생성하지 않고, 상위 트라이얼 run에
    step 기반 메트릭으로 기록합니다 (MLflow UI 성능 보호).
    """
    try:
        import mlflow
        mlflow.log_metric(f"file_{metric_name}", score, step=step)
    except Exception as e:
        logger.warning(f"MLflow 파일 메트릭 기록 실패 (비치명적): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna 하이퍼파라미터 자동 튜닝",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━ 최적화 전략 조합 예시 ━━━

  ⚡ 빠른 탐색 (10% 샘플 + 프루닝 + 최대 5곡):
    python -m ai_core.tune --source ~/midi/ --target keyboard \\
        --n-trials 100 --pruning --sample-ratio 0.1 --max-files 5

  🔍 정밀 검증 (전체 데이터 + 프루닝):
    python -m ai_core.tune --source ~/midi/ --target keyboard \\
        --n-trials 30 --pruning

  📊 대시보드로 실시간 모니터링:
    optuna-dashboard sqlite:///optuna_studies.db
""",
    )

    # ── 필수 옵션 ──
    parser.add_argument("--source", required=True,
                        help="소스 MIDI (파일 또는 디렉토리)")
    parser.add_argument("--target", required=True,
                        help="타겟 악기 (keyboard, violin, etc.)")

    # ── 기본 옵션 ──
    parser.add_argument("--genre", default="CLASSICAL",
                        help="장르 (기본: CLASSICAL)")
    parser.add_argument("--model-type", default=None,
                        help="모델 타입 (기본: registry 기본값)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna 탐색 횟수 (기본: 30)")
    parser.add_argument("--optimize", default="chord_accuracy",
                        choices=list(METRIC_DIRECTIONS.keys()),
                        help="최적화 대상 메트릭 (기본: chord_accuracy)")
    parser.add_argument("--direction", default=None,
                        choices=["maximize", "minimize"],
                        help="최적화 방향 (기본: 메트릭에 따라 자동)")

    # ── 최적화 전략 옵션 ──
    strategy = parser.add_argument_group(
        "최적화 전략",
        "튜닝 속도를 높이는 세 가지 전략을 독립적으로 또는 조합하여 사용"
    )
    strategy.add_argument(
        "--pruning", action="store_true",
        help="조기 종료 활성화: 중간 결과가 나쁘면 해당 트라이얼 즉시 중단. "
             "파일 5개 이상일 때 효과적. (장점: 시간 절약, 단점: 초반 파일에 편향 가능)"
    )
    strategy.add_argument(
        "--sample-ratio", type=float, default=1.0, metavar="N",
        help="데이터 샘플링 비율 (0.0~1.0). 예: 0.1 = 전체의 10%%만 사용. "
             "트라이얼마다 다른 서브셋 추출. (장점: 극적 속도 향상, 단점: 편향 위험) "
             "(기본: 1.0 = 전체 사용)"
    )
    strategy.add_argument(
        "--max-files", type=int, default=None, metavar="N",
        help="트라이얼당 최대 평가 파일 수. 예: 5 = 최대 5곡만 평가. "
             "(장점: 예측 가능한 실행 시간, 단점: 일반화 약함) "
             "(기본: 제한 없음)"
    )

    # ── 저장/기록 옵션 ──
    parser.add_argument("--study-name", default=None,
                        help="Optuna 스터디 이름 (기존 스터디 이어서 탐색)")
    parser.add_argument("--experiment", default="optuna-tuning",
                        help="MLflow 실험 이름")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH,
                        help=f"Optuna SQLite DB 경로 (기본: {DEFAULT_DB_PATH})")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="MLflow 기록 비활성화")

    args = parser.parse_args()

    # ── 입력 검증 ──
    if args.sample_ratio <= 0.0 or args.sample_ratio > 1.0:
        parser.error("--sample-ratio는 0.0 초과 1.0 이하여야 합니다")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files는 1 이상이어야 합니다")

    # ── 의존성 확인 ──
    try:
        import optuna
    except ImportError as e:
        logger.error(f"필수 패키지 없음: {e}")
        logger.error("설치: uv sync --extra gpu --extra mlflow")
        sys.exit(1)

    if not args.no_mlflow:
        try:
            import mlflow
        except ImportError:
            logger.warning("MLflow가 없습니다. --no-mlflow 모드로 전환합니다.")
            args.no_mlflow = True

    # ── 모델 로드 (최초 1회) ──
    logger.info("🔧 모델 로딩 시작...")
    loaded = _load_model(args.model_type)

    # ── 소스 파일 수집 ──
    source_files = _collect_source_files(args.source)
    logger.info(f"📁 소스 MIDI: {len(source_files)}개")

    # ── 최적화 방향 ──
    direction = args.direction or METRIC_DIRECTIONS[args.optimize]
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"tune-{args.target}-{args.optimize}-{timestamp}"

    # ── SQLite 저장소 (optuna-dashboard 연동) ──
    db_path = Path(args.db_path).resolve()
    storage = f"sqlite:///{db_path}"

    # ── 임시 출력 디렉토리 ──
    output_dir = Path(tempfile.mkdtemp(prefix="optuna_tune_"))

    # ── MLflow 설정 (연결 실패 시 자동 비활성화) ──
    if not args.no_mlflow:
        try:
            import mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(args.experiment)
            logger.info(f"MLflow 연결 성공: {tracking_uri}")
        except Exception as e:
            logger.warning(f"MLflow 연결 실패, 기록 없이 진행합니다: {e}")
            args.no_mlflow = True

    # ── Optuna 프루너 설정 ──
    # MedianPruner: 중간 결과가 전체 트라이얼 중간값보다 나쁘면 프루닝
    #
    # n_startup_trials=3: 처음 3개 트라이얼은 프루닝 없이 완주시켜 기준 축적.
    #   → 초기 데이터가 부족하면 프루너가 너무 공격적으로 잘라낼 수 있음.
    #
    # n_warmup_steps=1: 각 트라이얼에서 최소 1곡은 평가한 후 프루닝 판단.
    #   → 첫 곡이 운 나쁘게 나빠도 2곡째까지는 기회를 줌.
    #
    # interval_steps=1: 매 1 step(곡)마다 프루닝 체크.
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=3,   # 최소 3개 트라이얼은 프루닝 없이 기준 축적
            n_warmup_steps=1,     # 각 트라이얼에서 최소 1곡은 평가 후 판단
            interval_steps=1,     # 매 곡마다 프루닝 체크
        )
        if args.pruning
        else optuna.pruners.NopPruner()  # 프루닝 비활성화 시
    )

    # ── Optuna 스터디 생성 (SQLite 영속 저장) ──
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    # ── 실행 정보 표시 ──
    # 예상 소요 시간 계산
    effective_files = len(source_files)
    if args.sample_ratio < 1.0:
        effective_files = max(1, int(effective_files * args.sample_ratio))
    if args.max_files is not None:
        effective_files = min(effective_files, args.max_files)

    logger.info("═" * 60)
    logger.info("  🎯 Optuna 하이퍼파라미터 튜닝")
    logger.info(f"  Target     : {args.target}")
    logger.info(f"  Genre      : {args.genre}")
    logger.info(f"  Optimize   : {args.optimize} ({direction})")
    logger.info(f"  Trials     : {args.n_trials}")
    logger.info(f"  Source MIDI: {len(source_files)}개")
    logger.info(f"  Study      : {study_name}")
    logger.info(f"  DB         : {db_path}")
    # ── 최적화 전략 표시 ──
    if args.pruning:
        logger.info("  ✂ Pruning  : 활성 (MedianPruner)")
    if args.sample_ratio < 1.0:
        logger.info(f"  📊 Sampling : {args.sample_ratio:.0%} "
                     f"(~{effective_files}곡/트라이얼)")
    if args.max_files is not None:
        logger.info(f"  📏 Max Files: {args.max_files}곡/트라이얼")
    if effective_files < len(source_files):
        est_min = effective_files * 0.5 * args.n_trials  # 곡당 ~30초 기준
        est_max = effective_files * 1.0 * args.n_trials  # 곡당 ~60초 기준
        logger.info(f"  ⏱ 예상 시간 : {est_min:.0f}~{est_max:.0f}분")
    logger.info(f"  Dashboard  : optuna-dashboard sqlite:///{db_path}")
    logger.info("═" * 60)

    # ── objective 생성 ──
    objective = create_objective(
        source_files, args.target, args.genre, loaded, args.optimize, output_dir,
        enable_pruning=args.pruning,
        sample_ratio=args.sample_ratio,
        max_files=args.max_files,
    )

    # ── 최적화 실행 ──
    try:
        if not args.no_mlflow:
            import mlflow
            with mlflow.start_run(run_name=f"tuning-{study_name}"):
                mlflow.log_param("target", args.target)
                mlflow.log_param("genre", args.genre)
                mlflow.log_param("optimize_metric", args.optimize)
                mlflow.log_param("direction", direction)
                mlflow.log_param("n_trials", args.n_trials)
                mlflow.log_param("n_source_files", len(source_files))
                mlflow.log_param("db_path", str(db_path))
                # 최적화 전략 파라미터도 MLflow에 기록
                mlflow.log_param("pruning", args.pruning)
                mlflow.log_param("sample_ratio", args.sample_ratio)
                mlflow.log_param("max_files", args.max_files or "all")
                mlflow.log_param("effective_files_per_trial", effective_files)

                study.optimize(objective, n_trials=args.n_trials)

                # 최적 결과 기록
                if study.best_trial:
                    mlflow.log_metric(f"best_{args.optimize}", study.best_value)
                    for key, value in study.best_params.items():
                        mlflow.log_metric(f"best_{key}", value)
        else:
            study.optimize(objective, n_trials=args.n_trials)
    finally:
        # 임시 출력 디렉토리 정리
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info(f"임시 디렉토리 정리 완료: {output_dir}")

    # ── 결과 출력 ──
    # 프루닝 통계
    n_pruned = len([t for t in study.trials
                    if t.state.name == "PRUNED"])
    n_complete = len([t for t in study.trials
                      if t.state.name == "COMPLETE"])
    n_fail = len(study.trials) - n_pruned - n_complete

    print()
    print("═" * 60)
    print("  🏆 최적 하이퍼파라미터")
    print("═" * 60)

    if n_pruned > 0:
        print(f"  📊 트라이얼 통계:")
        print(f"     완료: {n_complete}  |  프루닝: {n_pruned}  |  실패: {n_fail}")
        saved_pct = (n_pruned / max(1, n_pruned + n_complete)) * 100
        print(f"     → 프루닝으로 ~{saved_pct:.0f}% 시간 절약!")
        print()

    if study.best_trial:
        print(f"  Trial #{study.best_trial.number}")
        print(f"  {args.optimize}: {study.best_value:.4f}")
        print()
        for key, value in study.best_params.items():
            print(f"  {key:>15}: {value}")
        print()
        print("  → arrangement.py에 아래 값을 적용하세요:")
        print()
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"    {key:>15} = {value:.2f}")
            else:
                print(f"    {key:>15} = {value}")
    else:
        print("  유효한 트라이얼이 없습니다.")

    print()
    print(f"  📊 대시보드: optuna-dashboard sqlite:///{db_path}")
    print(f"     → http://localhost:8080 에서 결과를 시각화할 수 있습니다")
    print("═" * 60)


if __name__ == "__main__":
    main()
