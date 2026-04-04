from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from hybrid_quant.core import FeatureSnapshot, MarketBar, MarketDataBatch, StrategyContext, StrategySignal
from hybrid_quant.data import (
    DownloadRequest,
    HistoricalDataIngestionService,
    TemporalDatasetSplitter,
    TemporalSplitConfig,
)

if TYPE_CHECKING:
    from hybrid_quant.bootstrap import TradingApplication


@dataclass(slots=True)
class EpisodeData:
    name: str
    frame: pd.DataFrame
    bars: tuple[MarketBar, ...]
    features: tuple[FeatureSnapshot, ...]
    candidate_signals: tuple[StrategySignal, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self):
        return self.bars[0].timestamp if self.bars else None

    @property
    def end(self):
        return self.bars[-1].timestamp if self.bars else None


@dataclass(slots=True)
class EpisodeDataset:
    episodes: dict[str, EpisodeData]

    def get(self, name: str) -> EpisodeData:
        if name not in self.episodes:
            raise KeyError(f"Unknown episode split: {name}")
        return self.episodes[name]


@dataclass(slots=True)
class RLEpisodeBuilder:
    application: TradingApplication
    data_service: HistoricalDataIngestionService
    splitter: TemporalDatasetSplitter = field(default_factory=TemporalDatasetSplitter)

    def prepare_frame(
        self,
        *,
        request: DownloadRequest | None = None,
        input_frame: pd.DataFrame | None = None,
        allow_gaps: bool = False,
    ) -> pd.DataFrame:
        if input_frame is not None:
            cleaned_frame, _ = self.data_service.cleaner.clean(input_frame)
            self.data_service.validator.validate(
                cleaned_frame,
                self.application.settings.market.execution_timeframe,
                allow_gaps=allow_gaps,
            )
            return cleaned_frame
        if request is None:
            raise ValueError("RLEpisodeBuilder.prepare_frame requires a download request or an input frame.")
        frame, _, _ = self.data_service.prepare_frame(request=request, allow_gaps=allow_gaps)
        return frame

    def build_dataset(self, frame: pd.DataFrame, split_config: TemporalSplitConfig | None = None) -> EpisodeDataset:
        config = split_config or TemporalSplitConfig(
            train_ratio=self.application.settings.data.train_ratio,
            validation_ratio=self.application.settings.data.validation_ratio,
            test_ratio=self.application.settings.data.test_ratio,
        )
        split_frames = self.splitter.split(frame, config)
        episodes = {
            name: self._build_episode(name, dataset_split.frame)
            for name, dataset_split in split_frames.items()
            if not dataset_split.frame.empty
        }
        return EpisodeDataset(episodes=episodes)

    def _build_episode(self, name: str, frame: pd.DataFrame) -> EpisodeData:
        bars = self._frame_to_bars(frame)
        feature_snapshots = tuple(
            self.application.feature_pipeline.transform(
                MarketDataBatch(
                    symbol=self.application.settings.market.symbol,
                    timeframe=self.application.settings.market.execution_timeframe,
                    bars=bars,
                    metadata={"source": "rl_episode_builder", "split": name},
                )
            )
        )
        candidate_signals = tuple(self._generate_signals(bars, list(feature_snapshots)))
        return EpisodeData(
            name=name,
            frame=frame.copy(),
            bars=tuple(bars),
            features=feature_snapshots,
            candidate_signals=candidate_signals,
            metadata={
                "rows": len(frame),
                "split": name,
                "symbol": self.application.settings.market.symbol,
            },
        )

    def _frame_to_bars(self, frame: pd.DataFrame) -> list[MarketBar]:
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index, utc=True)
        normalized = normalized.sort_index(kind="mergesort")
        return [
            MarketBar(
                timestamp=timestamp.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for timestamp, row in normalized.iterrows()
        ]

    def _generate_signals(
        self,
        bars: list[MarketBar],
        feature_snapshots: list[FeatureSnapshot],
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for bar, feature in zip(bars, feature_snapshots, strict=True):
            adx = feature.values.get("adx_1h")
            regime = (
                "trend"
                if adx is not None
                and math.isfinite(float(adx))
                and float(adx) > self.application.settings.strategy.adx_threshold
                else "range"
            )
            signals.append(
                self.application.strategy.generate(
                    StrategyContext(
                        symbol=self.application.settings.market.symbol,
                        execution_timeframe=self.application.settings.market.execution_timeframe,
                        filter_timeframe=self.application.settings.market.filter_timeframe,
                        bars=[bar],
                        features=[feature],
                        regime=regime,
                    )
                )
            )
        return signals
