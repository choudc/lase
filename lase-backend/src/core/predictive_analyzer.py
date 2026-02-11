from datetime import datetime, timedelta, timezone

from ..models.session import Task


class PredictiveAnalyzer:
    """
    Feasible (non-ML) predictor: rule-based warnings derived from recent task history.
    This is meant to match the design doc’s "medium feasibility" scope.
    """

    def predict(self) -> dict:
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=24)

        recent = Task.query.filter(Task.created_at >= since).all()
        if not recent:
            return {"status": "ok", "window_hours": 24, "predictions": []}

        failures = [t for t in recent if t.status == "failed"]
        running = [t for t in recent if t.status == "running"]

        predictions = []
        if len(failures) >= 3:
            predictions.append(
                {
                    "type": "stability_risk",
                    "severity": "medium" if len(failures) < 6 else "high",
                    "message": f"{len(failures)} tasks failed in the last 24h",
                    "suggestion": "Run /api/quality/audit and inspect timeline logs for repeated errors",
                }
            )

        if len(running) >= 5:
            predictions.append(
                {
                    "type": "throughput_risk",
                    "severity": "low",
                    "message": f"{len(running)} tasks are currently running",
                    "suggestion": "Consider reducing concurrency or increasing resources",
                }
            )

        return {"status": "ok", "window_hours": 24, "predictions": predictions}

