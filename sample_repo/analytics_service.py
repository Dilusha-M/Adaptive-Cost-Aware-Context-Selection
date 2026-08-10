"""Analytics service for tracking application metrics."""
from base_model import BaseModel


class AnalyticsService:
    """Collects and analyzes application usage data."""

    def __init__(self):
        self._events = []
        self._metrics = {}

    def track_event(self, event_type: str, data: dict) -> None:
        """Record an application event."""
        self._events.append({"type": event_type, "data": data})

    def get_metric(self, metric_name: str) -> float:
        """Retrieve a named metric value."""
        return self._metrics.get(metric_name, 0.0)

    def update_metric(self, metric_name: str, value: float) -> None:
        """Update a metric value."""
        self._metrics[metric_name] = value

    def get_event_log(self) -> list:
        """Return the full event log."""
        return list(self._events)

    def generate_report(self) -> dict:
        """Generate an analytics report."""
        return {
            "total_events": len(self._events),
            "metrics": dict(self._metrics)
        }


class ReportModel(BaseModel):
    """Model for storing generated reports."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content = None
        self.generated_at = None
