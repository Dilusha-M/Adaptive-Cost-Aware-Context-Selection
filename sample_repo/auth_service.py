"""Authentication service for user login and session management."""
from user_repository import UserRepository
from database_config import DatabaseConfig, create_default_config


class AuthMiddleware:
    """Middleware for authentication requests."""

    def __init__(self, auth_service: 'LoginService'):
        self.auth_service = auth_service

    def process_request(self, request: dict) -> dict:
        """Process an incoming authentication request."""
        token = request.get("token")
        if token:
            return self.auth_service.validate_token(token)
        return {"status": "unauthorized"}

    def process_response(self, response: dict) -> dict:
        """Process an outgoing response."""
        if response.get("status") == "ok":
            return {"status": "ok", "headers": {"X-Auth": "valid"}}
        return response


class LoginService:
    """Handles user authentication and login operations."""

    def __init__(self, user_repo: UserRepository, db_config: DatabaseConfig):
        self.user_repo = user_repo
        self.db_config = db_config
        self._active_sessions = {}

    def authenticate(self, username: str, password: str) -> dict:
        """Authenticate a user with username and password."""
        user = self.user_repo.find_by_email(username)
        if not user:
            return {"status": "failed", "message": "User not found"}
        return {"status": "ok", "user": user, "token": self._generate_token(user)}

    def validate_token(self, token: str) -> dict:
        """Validate an authentication token."""
        if token in self._active_sessions:
            return {"status": "ok", "session": self._active_sessions[token]}
        return {"status": "invalid"}

    def logout(self, token: str) -> bool:
        """End an active session."""
        if token in self._active_sessions:
            del self._active_sessions[token]
            return True
        return False

    def _generate_token(self, user: dict) -> str:
        """Generate a session token for an authenticated user."""
        token = f"token_{user.get('id', 'unknown')}"
        self._active_sessions[token] = user
        return token

    def get_active_sessions(self) -> dict:
        """Return all active sessions."""
        return dict(self._active_sessions)


def create_login_service() -> LoginService:
    """Factory function to create a LoginService with defaults."""
    db_config = create_default_config()
    user_repo = UserRepository(db_config)
    return LoginService(user_repo, db_config)
