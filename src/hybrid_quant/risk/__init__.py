__all__ = [
    "AzirRiskConfig",
    "AzirRiskDecision",
    "AzirRiskEngine",
    "AzirRiskState",
    "PropFirmRiskEngine",
    "RiskEngine",
    "evaluate_anomalies",
    "run_anomaly_evaluation",
]


def __getattr__(name: str):
    if name in {"AzirRiskConfig", "AzirRiskDecision", "AzirRiskState"}:
        from .azir_state import AzirRiskConfig, AzirRiskDecision, AzirRiskState

        return {
            "AzirRiskConfig": AzirRiskConfig,
            "AzirRiskDecision": AzirRiskDecision,
            "AzirRiskState": AzirRiskState,
        }[name]
    if name in {"AzirRiskEngine", "evaluate_anomalies", "run_anomaly_evaluation"}:
        from .azir_engine import AzirRiskEngine, evaluate_anomalies, run_anomaly_evaluation

        return {
            "AzirRiskEngine": AzirRiskEngine,
            "evaluate_anomalies": evaluate_anomalies,
            "run_anomaly_evaluation": run_anomaly_evaluation,
        }[name]
    if name in {"PropFirmRiskEngine", "RiskEngine"}:
        from .engine import PropFirmRiskEngine, RiskEngine

        return {"PropFirmRiskEngine": PropFirmRiskEngine, "RiskEngine": RiskEngine}[name]
    raise AttributeError(f"module 'hybrid_quant.risk' has no attribute {name!r}")
