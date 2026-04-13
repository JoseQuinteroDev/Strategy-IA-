__all__ = [
    "AzirEventReplayEnvironment",
    "AzirReplayEvent",
    "HybridTradingEnvironment",
    "TradingEnvironment",
    "build_azir_event_replay_dataset",
]


def __getattr__(name: str):
    if name in {"AzirEventReplayEnvironment", "AzirReplayEvent", "build_azir_event_replay_dataset"}:
        from .azir_event_env import AzirEventReplayEnvironment, AzirReplayEvent, build_azir_event_replay_dataset

        return {
            "AzirEventReplayEnvironment": AzirEventReplayEnvironment,
            "AzirReplayEvent": AzirReplayEvent,
            "build_azir_event_replay_dataset": build_azir_event_replay_dataset,
        }[name]
    if name in {"HybridTradingEnvironment", "TradingEnvironment"}:
        from .environment import HybridTradingEnvironment, TradingEnvironment

        return {
            "HybridTradingEnvironment": HybridTradingEnvironment,
            "TradingEnvironment": TradingEnvironment,
        }[name]
    raise AttributeError(f"module 'hybrid_quant.env' has no attribute {name!r}")
