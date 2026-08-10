"""Database configuration and connection management."""
from typing import Optional


class DatabaseConfig:
    """Configuration for database connections."""

    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self._connection = None

    def get_connection_string(self) -> str:
        """Build a connection string from configuration."""
        return f"postgresql://{self.host}:{self.port}/{self.database}"

    def initialize(self) -> bool:
        """Initialize the database connection."""
        self._connection = self.get_connection_string()
        return True

    def close(self) -> None:
        """Close the database connection."""
        self._connection = None

    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connection is not None


def create_default_config() -> DatabaseConfig:
    """Create a default database configuration."""
    return DatabaseConfig(host="localhost", port=5432, database="app_db")
