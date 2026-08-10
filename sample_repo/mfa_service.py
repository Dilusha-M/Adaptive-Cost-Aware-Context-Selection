"""Multi-factor authentication service."""
import hashlib
import random
from auth_service import LoginService
from user_repository import UserRepository


class MFAService:
    """Handles multi-factor authentication flows."""

    def __init__(self, auth_service: LoginService, user_repo: UserRepository):
        self.auth_service = auth_service
        self.user_repo = user_repo
        self._mfa_secrets = {}
        self._verification_codes = {}

    def enable_mfa(self, user_id: int) -> dict:
        """Enable MFA for a user."""
        user = self.user_repo.find_by_id(user_id)
        if not user:
            return {"status": "failed", "message": "User not found"}
        secret = self._generate_secret()
        self._mfa_secrets[user_id] = secret
        return {"status": "enabled", "user_id": user_id, "secret": secret}

    def disable_mfa(self, user_id: int) -> bool:
        """Disable MFA for a user."""
        self._mfa_secrets.pop(user_id, None)
        return True

    def verify_mfa(self, user_id: int, code: str) -> bool:
        """Verify an MFA verification code."""
        stored_code = self._verification_codes.get(user_id)
        if stored_code == code:
            del self._verification_codes[user_id]
            return True
        return False

    def generate_code(self, user_id: int) -> str:
        """Generate a one-time verification code."""
        code = str(random.randint(100000, 999999))
        self._verification_codes[user_id] = code
        return code

    def _generate_secret(self) -> str:
        """Generate a random MFA secret."""
        return hashlib.sha256(str(random.random()).encode()).hexdigest()
