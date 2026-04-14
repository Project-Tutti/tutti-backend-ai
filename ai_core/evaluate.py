"""ai_core/evaluate.py — 편곡 품질 평가 + MLflow 자동 기록.

AI 개발자가 모델 테스트할 때 사용하는 스크립트.
결과가 MLflow에 자동 기록됩니다.

사용법:
    # 단일 파일 평가
    python -m ai_core.evaluate \
        --source INPUT/KissTheRain.mid \
        --generated OUTPUT/KissTheRain_violin.mid

    # 디렉토리 배치 평가
    python -m ai_core.evaluate \
        --source INPUT/ \
        --generated OUTPUT/ \
        --experiment "violin-v2-test"

    # MLflow 기록 끄기
    python -m ai_core.evaluate \
        --source INPUT/test.mid \
        --generated OUTPUT/test.mid \
        --no-mlflow
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from ai_core.metrics import compute_basic_quality_metrics, compute_musical_quality

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 결과 출력용 음정 이름
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def evaluate_pair(source_path: str, generated_path: str,
                  target_program: int | None = None) -> dict:
    """원본-생성 MIDI 쌍을 평가하고 전체 메트릭을 반환합니다."""

    # Stage 1: 기본 통계
    basic = compute_basic_quality_metrics(generated_path)

    # Stage 2: 음악적 평가
    musical = compute_musical_quality(
        source_path=source_path,
        generated_path=generated_path,
        target_program=target_program,
    )

    result = {}
    result.update(basic)
    result.update(musical)
    return result


def print_report(source: str, generated: str, metrics: dict):
    """터미널에 평가 결과를 표시합니다."""
    print()
    print(f"📊 음악적 평가")
    print(f"   소스:    {Path(source).name}")
    print(f"   생성:    {Path(generated).name}")
    print()
    print(f"┌─────────────────────────────────────────────────────┐")
    print(f"│  기본 통계                                          │")
    print(f"├─────────────────────────────────────────────────────┤")
    print(f"│  노트 수             {metrics.get('note_count', '-'):>8}                  │")
    print(f"│  음역 (반음)         {metrics.get('pitch_range', '-'):>8}                  │")
    print(f"│  평균 음정           {metrics.get('pitch_mean', '-'):>8}                  │")
    print(f"│  평균 벨로시티       {metrics.get('avg_velocity', '-'):>8}                  │")
    print(f"│  밀도 (노트/초)      {metrics.get('density_per_sec', '-'):>8}                  │")
    print(f"├─────────────────────────────────────────────────────┤")
    print(f"│  음악적 품질                         점수   (범위)  │")
    print(f"├─────────────────────────────────────────────────────┤")

    ca = metrics.get("chord_accuracy")
    ps = metrics.get("pch_similarity")
    doa = metrics.get("doa")
    dr = metrics.get("dissonance_rate")

    if ca is not None:
        print(f"│  Chord Accuracy          {ca:>8.4f}   (0~1, ↑좋음)  │")
        print(f"│  PCH Similarity          {ps:>8.4f}   (0~1, ↑좋음)  │")
        print(f"│  DOA (창의성)            {doa:>8.4f}   (0~1, ↑다양)  │")
        print(f"│  Dissonance Rate         {dr:>8.4f}   (0~1, ↓좋음)  │")
    else:
        print(f"│  (음악적 평가 사용 불가 — numpy/pretty_midi 필요)   │")

    print(f"└─────────────────────────────────────────────────────┘")


def log_to_mlflow(source: str, generated: str, metrics: dict,
                  experiment: str = "arrangement-eval",
                  run_name: str | None = None):
    """MLflow에 평가 결과를 기록합니다."""
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow가 설치되지 않아 기록을 건너뜁니다")
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment)

    if run_name is None:
        run_name = f"{Path(source).stem} → {Path(generated).stem}"

    with mlflow.start_run(run_name=run_name):
        # 파라미터
        mlflow.log_param("source_file", Path(source).name)
        mlflow.log_param("generated_file", Path(generated).name)

        # 메트릭
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # 결과 파일 첨부
        if Path(generated).exists():
            mlflow.log_artifact(generated, "generated_midi")

    logger.info(f"✅ MLflow 기록 완료 — 실험: {experiment}, 실행: {run_name}")


def main():
    parser = argparse.ArgumentParser(description="편곡 품질 평가 + MLflow 기록")
    parser.add_argument("--source", required=True, help="원본 소스 MIDI (파일 또는 디렉토리)")
    parser.add_argument("--generated", required=True, help="생성된 MIDI (파일 또는 디렉토리)")
    parser.add_argument("--target-program", type=int, default=None, help="타겟 악기 program 번호")
    parser.add_argument("--experiment", default="arrangement-eval", help="MLflow 실험 이름")
    parser.add_argument("--no-mlflow", action="store_true", help="MLflow 기록 비활성화")
    args = parser.parse_args()

    source_path = Path(args.source)
    generated_path = Path(args.generated)

    # 디렉토리면 배치 평가
    if source_path.is_dir() and generated_path.is_dir():
        pairs = []
        gen_files = sorted(generated_path.glob("*.mid")) + sorted(generated_path.glob("*.midi"))
        for gen_file in gen_files:
            src_file = source_path / gen_file.name
            if src_file.exists():
                pairs.append((str(src_file), str(gen_file)))
            else:
                logger.warning(f"소스 파일 없음, 건너뜀: {gen_file.name}")

        if not pairs:
            logger.error("평가할 MIDI 쌍을 찾을 수 없습니다")
            sys.exit(1)

        logger.info(f"배치 평가: {len(pairs)}쌍 발견")
        all_metrics = []

        for src, gen in pairs:
            metrics = evaluate_pair(src, gen, args.target_program)
            print_report(src, gen, metrics)
            all_metrics.append(metrics)

            if not args.no_mlflow:
                log_to_mlflow(src, gen, metrics, experiment=args.experiment)

        # 배치 평균 출력
        if all_metrics and all_metrics[0].get("chord_accuracy") is not None:
            print(f"\n📊 배치 평균 ({len(all_metrics)}건)")
            for key in ["chord_accuracy", "pch_similarity", "doa", "dissonance_rate"]:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg = sum(values) / len(values)
                    print(f"   {key}: {avg:.4f}")

    else:
        # 단일 파일 평가
        metrics = evaluate_pair(str(source_path), str(generated_path), args.target_program)
        print_report(str(source_path), str(generated_path), metrics)

        if not args.no_mlflow:
            log_to_mlflow(
                str(source_path), str(generated_path), metrics,
                experiment=args.experiment,
            )


if __name__ == "__main__":
    main()
