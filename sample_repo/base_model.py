"""Base models and common utilities for the application."""
from typing import Any, Dict, Optional


class BaseModel:
    """Base model with common functionality for all domain models."""

    def __init__(self, id: Optional[int] = None):
        self.id = id
        self._metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {"id": self.id, "metadata": self._metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create a model instance from a dictionary."""
        instance = cls(id=data.get("id"))
        instance._metadata = data.get("metadata", {})
        return instance


class TimestampMixin:
    """Mixin that adds timestamp tracking to models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = None
        self.updated_at = None

    def set_timestamps(self) -> None:
        """Update both created_at and updated_at timestamps."""
        import datetime
        now = datetime.datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now
