from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hybrid_quant.data import read_ohlcv_frame

from .diagnostics import BaselineDiagnosticsArtifacts, BaselineDiagnosticsRunner
from .runner import BaselineArtifacts, BaselineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a baseline from input-path and immediately generate diagnostics."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--variant", default="baseline_trend_nasdaq")
    parser.add_argument("--input-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-gaps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else output_dir / "baseline"
    diagnostics_dir = output_dir / "diagnostics"

    if args.input_path:
        runner = BaselineRunner.from_config(args.config_dir, variant_name=args.variant)
        frame = _read_input_frame(args.input_path)
        baseline_artifacts = runner.run(
            output_dir=artifact_dir,
            input_frame=frame,
            allow_gaps=args.allow_gaps or runner.application.settings.data.allow_gaps,
        )
    else:
        baseline_artifacts = None
        if not artifact_dir.exists():
            raise ValueError("Provide either --input-path or an existing --artifact-dir.")

    diagnostics_runner = BaselineDiagnosticsRunner.from_config(
        args.config_dir,
        variant_name=args.variant,
    )
    diagnostics_artifacts = diagnostics_runner.run(
        artifact_dir=artifact_dir,
        output_dir=diagnostics_dir,
    )

    payload = json.loads(diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8"))
    print(f"Baseline artifacts: {artifact_dir}")
    print(f"Diagnostics report: {diagnostics_artifacts.diagnostics_path}")
    print(f"Diagnostics summary: {diagnostics_artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variant={args.variant}",
                f"trades={payload['baseline_metrics']['number_of_trades']}",
                f"net_pnl={payload['baseline_metrics']['net_pnl']}",
                f"validation_readiness={payload['automatic_conclusion']['validation_verdict']}",
            ]
        )
    )
    return 0


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    return read_ohlcv_frame(path)


if __name__ == "__main__":
    raise SystemExit(main())
