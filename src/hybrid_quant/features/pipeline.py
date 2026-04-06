from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hybrid_quant.core import FeatureSnapshot, MarketDataBatch

from .deterministic import DeterministicFeatureConfig, build_features


@dataclass(slots=True)
class FeaturePipeline:
    feature_names: list[str]
    lookback_window: int
    regime_window: int
    normalize: bool = True
    deterministic_config: DeterministicFeatureConfig | None = None

    def transform(self, batch: MarketDataBatch) -> list[FeatureSnapshot]:
        if not batch.bars:
            return []

        frame = pd.DataFrame(
            [
                {
                    "open_time": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in batch.bars
            ]
        )
        feature_frame = build_features(frame, config=self.deterministic_config)
        selected_frame = feature_frame
        if self.feature_names:
            missing = [name for name in self.feature_names if name not in feature_frame.columns]
            if missing:
                raise ValueError(f"Unknown features requested from pipeline: {missing}")
            selected_frame = feature_frame[self.feature_names]

        snapshots: list[FeatureSnapshot] = []
        for timestamp, row in selected_frame.iterrows():
            values = {
                name: float(value) if pd.notna(value) else float("nan")
                for name, value in row.items()
            }
            snapshots.append(
                FeatureSnapshot(
                    timestamp=timestamp.to_pydatetime(),
                    values=values,
                    metadata={
                        "normalize": self.normalize,
                        "lookback_window": self.lookback_window,
                        "regime_window": self.regime_window,
                        "source": "deterministic",
                    },
                )
            )
        return snapshots
